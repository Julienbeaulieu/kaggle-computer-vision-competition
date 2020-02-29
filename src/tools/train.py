"""

train model
Usage:
    train.py -o=<path> [--data_cfg=<path>]  [--cfg=<path>]
    train.py -h | --help

Options:
    -h --help               show this screen help
    -o=<path>               output path
    --data_cfg=<path>       data config path [default: configs/data.yaml]
    --cfg=<path>            training config path
"""

import os
import time
import json
import pickle
import torch
import torchvision
import numpy as np
from pathlib import Path
from docopt import docopt
from datetime import datetime
from src.modeling.meta_arch.build import build_model
from src.data.bengali_data import build_data_loader
from src.modeling.solver import build_optimizer, build_scheduler, build_evaluator, MixupAugmenter

# import configurations
from yacs.config import CfgNode
from src.config import get_cfg_defaults

# For Tensorboard integration
from torch.utils.tensorboard import SummaryWriter


def train(cfg, debug=False):

    #############################
    # Pre-training shenanigans
    #############################

    # PATHS
    assert cfg.OUTPUT_PATH != ''
    output_path = cfg.OUTPUT_PATH # output folder
    train_path = cfg.DATASET.TRAIN_DATA_PATH # training data folder
    val_path = cfg.DATASET.VAL_DATA_PATH # validation data folder

    # sample is 1/4th of the train images - aka 1 .parquet file
    train_path_sample = cfg.DATASET.TRAIN_DATA_SAMPLE
    valid_path_sample = cfg.DATASET.VALID_DATA_SAMPLE

    # Create writable timestamp for easier record keeping
    timestamp = datetime.now().isoformat(sep="T", timespec="auto")
    name_timestamp = timestamp.replace(":", "_")

    # Make output dir and its parents if they do not exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Make backup folders if they do not exist
    backup_dir = os.path.join(output_path, 'model_backups')
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)
    
    # Make result folders if they do not exist 
    results_dir = os.path.join(output_path, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # to initialize Tensorboard
    writer_tensorboard = SummaryWriter(log_dir=results_dir + "logs_tensorflow")

    cfg.dump(stream=open(os.path.join(results_dir, f'config_{name_timestamp}.yaml'), 'w'))

    # file path to store the state of the model
    state_fpath = os.path.join(output_path, f'model_{name_timestamp}.pt')

    # Performance path where we'll save our metrics to trace.p
    perf_path = os.path.join(results_dir, f'trace_{name_timestamp}.p')
    perf_trace = []

    # debug: load a smaller training file
    if debug:
        train_data = pickle.load(open(train_path_sample, 'rb'))
        val_data = pickle.load(open(valid_path_sample, 'rb'))
    else:
        train_data = pickle.load(open(train_path, 'rb'))
        val_data = pickle.load(open(val_path, 'rb'))

    # DataLoader
    train_loader = build_data_loader(train_data, cfg.DATASET, True)
    val_loader = build_data_loader(val_data, cfg.DATASET, False)

    # Build model using config dict node
    model = build_model(cfg.MODEL)

    # For resuming training
    if cfg.RESUME_PATH != "":
        checkpoint = torch.load(cfg.RESUME_PATH, map_location='cpu')
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint["model_state"])
    model = model.cuda()

    # Solver evaluator
    solver_cfg = cfg.MODEL.SOLVER

    # Epochs
    current_epoch = 0
    total_epochs = solver_cfg.SCHEDULER.TOTAL_EPOCHS

    # Build optimizer
    opti_cfg = solver_cfg.OPTIMIZER
    optimizer = build_optimizer(model, opti_cfg)

    # Build scheduler
    sched_cfg = solver_cfg.SCHEDULER
    scheduler = build_scheduler(optimizer, sched_cfg, steps_per_epoch=np.int(len(train_loader)), epochs=total_epochs)

    # Build evaluator with or without Mixup
    mixup_training = solver_cfg.MIXUP_AUGMENT
    if mixup_training:
        mixup_augmenter = MixupAugmenter(solver_cfg.MIXUP)
    evaluator, mixup_evaluator = build_evaluator(solver_cfg)
    evaluator.float().cuda()
    if mixup_evaluator is not None:
        mixup_evaluator.float().cuda()

    ##########################################
    # Main training epoch loop starts here   
    ##########################################
    for epoch in range(current_epoch, total_epochs):
        model.train()
        print('Start epoch', epoch)
        train_itr = iter(train_loader)
        total_err = 0
        total_acc = 0

        for idx, (inputs, labels) in enumerate(train_itr):

            # Compute
            input_data = inputs.float().cuda()
            labels = labels.cuda()

            # Use the model to produce the classification
            if mixup_training:
                input_data, labels = mixup_augmenter(input_data, labels)
            grapheme_logits, vowel_logits, consonant_logits = model(input_data)

            # Calling MultiHeadsEval forward function to produce evaluator results
            if mixup_training:
                eval_result = mixup_evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
            else:
                eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
            optimizer.zero_grad()

            # get loss, back propagate, step
            loss = eval_result['loss']
            loss.backward()
            optimizer.step()

            # tabulate the steps from the evaluation
            eval_result = {k: eval_result[k].item() for k in eval_result}        
            scheduler.step()

            if idx % 100 == 0:
                print(idx, eval_result['loss'], eval_result['acc'])
        
        ###############################
        # Send images to Tensorboard 
        # -- could also do this outside the loop with xb, yb = next(itr(DL))
        ###############################

        # Get the std and mean of each channel
        std = torch.FloatTensor(cfg.DATASET.NORMALIZE_STD).view(3,1,1)
        m = torch.FloatTensor(cfg.DATASET.NORMALIZE_STD).view(3,1,1)

        # Un-normalize images, send mean and std to gpu for mixuped images
        inputs, input_data = inputs*m, input_data*m.cuda()
        inputs, input_data = inputs*std, input_data*std.cuda()

        # Squeeze the RGB channels down to 1 channel
        #inputs, input_data = inputs.mean(1)[:,None,:,:], input_data.mean(1)[:,None,:,:]

        img_grid = torchvision.utils.make_grid(inputs)
        img_grid_mixup = torchvision.utils.make_grid(input_data)

        writer_tensorboard.add_image("images no mixup", img_grid)
        writer_tensorboard.add_image("images with mixup", img_grid_mixup)    
        
        ####################
        # Training metrics
        ####################
        if mixup_training:
            train_result = mixup_evaluator.evaluate_on_cache()
            mixup_evaluator.clear_cache()
        else:
            train_result = evaluator.evaluate_on_cache()

        # Store training loss, accuracy, kaggle score and write to Tensorboard
        train_total_err = train_result['loss']
        writer_tensorboard.add_scalar('Loss/train', train_total_err, global_step=epoch)

        train_total_acc = train_result['acc']
        writer_tensorboard.add_scalar('Accuracy/train', train_total_acc, global_step=epoch)

        train_kaggle_score = train_result['kaggle_score']
        writer_tensorboard.add_scalar('Kaggle_Score/train', train_kaggle_score, global_step=epoch)

        conv2d = model.backbone.features[0][0].weight
        conv2d_grad = model.backbone.features[0][0].weight.grad
        conv2d_convBNReLU = model.backbone.features[1].conv[0][0].weight
        conv2d_convBNReLU_grad = model.backbone.features[1].conv[0][0].weight.grad

        writer_tensorboard.add_histogram('conv2d.weight', conv2d, epoch)
        writer_tensorboard.add_histogram('conv2d.weight.grad', conv2d_grad, epoch)
        writer_tensorboard.add_histogram('conv2d_BNReLU.weight', conv2d_convBNReLU, epoch)
        writer_tensorboard.add_histogram('conv2d_BNReLU.weight.grad', conv2d_convBNReLU_grad, epoch)

        # Print results
        print("Epoch {0} Training, Loss {1}, Acc {2}, kaggle Score {3}".format(epoch, train_total_err, train_total_acc, train_kaggle_score))
        evaluator.clear_cache()

        ###############################
        # Compute validation error
        ###############################
        model.eval()
        val_itr = iter(val_loader)
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_itr):
                input_data = inputs.float().cuda()
                labels = labels.cuda()
                grapheme_logits, vowel_logits, consonant_logits = model(input_data)
                eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
                eval_result = {k: eval_result[k].item() for k in eval_result}
                total_err += eval_result['loss']
                total_acc += eval_result['acc']
                # print(total_err / (1 + idx), total_acc / (1 + idx))

        #######################
        # Validation metrics
        #######################
        val_result = evaluator.evaluate_on_cache()

        # Store validation loss, accuracy, kaggle score and write to Tensorboard
        val_total_err = val_result['loss']
        writer_tensorboard.add_scalar('Loss/Val', val_total_err, global_step=epoch)

        val_total_acc = val_result['acc']
        writer_tensorboard.add_scalar('Accuracy/Val', val_total_acc, global_step=epoch)

        val_kaggle_score = val_result['kaggle_score']
        writer_tensorboard.add_scalar('Kaggle_Score/Val', val_kaggle_score, global_step=epoch)

        # track learning rate because we used OneCycleLR scheduler
        lr = optimizer.param_groups[-1]['lr']
        writer_tensorboard.add_scalar('learning_rate', lr, global_step=epoch)

        # Write to disk
        writer_tensorboard.flush()

        print("Epoch {0} Eval, Loss {1}, Acc {2}, Kaggle score {3}".format(epoch, val_total_err, val_total_acc, val_kaggle_score))
        evaluator.clear_cache()

        ######################################
        # Saving the model + performance
        ######################################
        print("Saving the model (epoch %d)" % epoch)
        
        # create save_state dict with all hyperparamater + parameters
        save_state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
                     }

        # if scheduler, add it to the save dict
        if scheduler is not None:
            save_state['scheduler_state'] = scheduler.state_dict()
        # save model
        torch.save(save_state, state_fpath)

        # save a backup
        print("Making a backup (step %d)" % epoch)
        backup_fpath = os.path.join(backup_dir, "model_bak_%06d.pt" % (epoch,))
        torch.save(save_state, backup_fpath)

        # Dump the traces
        perf_trace.append(
            {
                'epoch': epoch,
                'train_err': train_total_err,
                'train_acc': train_total_acc,
                'train_kaggle_score': train_kaggle_score,
                'val_err': val_total_err,
                'val_acc': val_total_acc,
                'val_kaggle_score': val_kaggle_score,
                'lr': lr
            }
        )
        pickle.dump(perf_trace, open(perf_path, 'wb'))

        # store epoch results separately
        epoch_results = {
            'epoch': epoch,
            'train_result': train_result,
            'val_result': val_result
        }   
        pickle.dump(epoch_results, open(os.path.join(results_dir, 'result_epoch_{0}.p'.format(epoch)), 'wb'))
    
    # Add model to Tensorboard to inspect the details of the architecture
    writer_tensorboard.add_graph(model, input_data)
    writer_tensorboard.close()

if __name__ == '__main__':

    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    output_path = arguments['-o']
    data_path = arguments['--data_cfg']
    cfg_path = arguments['--cfg']
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    if cfg_path is not None:
        cfg.merge_from_file(cfg_path)
    cfg.OUTPUT_PATH = output_path
    train(cfg)
