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


def train(cfg):

    #############################
    # Pre-training shenanigans
    #############################

    # PATHS
    assert cfg.OUTPUT_PATH != ''
    output_path = cfg.OUTPUT_PATH # output folder
    all_train_path = cfg.DATASET.ALL_DATA # training data folder

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


    all_train_data = pickle.load(open(all_train_path, 'rb'))
    
    # DataLoader
    all_train_loader = build_data_loader(all_train_data, cfg.DATASET, True)

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
    if cfg.RESUME_PATH == "":
        current_epoch = 0 
    total_epochs = solver_cfg.SCHEDULER.TOTAL_EPOCHS

    # Build optimizerW
    opti_cfg = solver_cfg.OPTIMIZER
    optimizer = build_optimizer(model, opti_cfg)

    # Build scheduler
    sched_cfg = solver_cfg.SCHEDULER
    scheduler_type = sched_cfg.NAME
    scheduler = build_scheduler(optimizer, sched_cfg, steps_per_epoch=np.int(len(all_train_loader)), epochs=total_epochs)

    # Resume training with correct optimizer and scheduler
    if cfg.RESUME_PATH != "" and sched_cfg.PROG_RESIZE is False:
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state'])

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
        all_train_itr = iter(all_train_loader)
        total_err = 0
        total_acc = 0

        for idx, (inputs, labels) in enumerate(all_train_itr):

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

            if scheduler_type == 'OneCycleLR':
                scheduler.step()       

            if idx % 100 == 0:
                print(idx, eval_result['loss'], eval_result['acc'])

        ###############################
        # Send images to Tensorboard 
        # -- could also do this outside the loop with xb, yb = next(itr(DL))
        ###############################

        if epoch == 0: 
            # Get the std and mean of each channel
            std = torch.FloatTensor(cfg.DATASET.NORMALIZE_STD).view(3,1,1)
            m = torch.FloatTensor(cfg.DATASET.NORMALIZE_MEAN).view(3,1,1)

            # Un-normalize images, send mean and std to gpu for mixuped images
            imgs, imgs_mixup = ((inputs*std)+m)*255, ((input_data*std.cuda())+m.cuda())*255
            imgs, imgs_mixup = imgs.type(torch.uint8), imgs_mixup.type(torch.uint8)
            img_grid = torchvision.utils.make_grid(imgs)
            img_grid_mixup = torchvision.utils.make_grid(imgs_mixup)

            img_grid = torchvision.utils.make_grid(imgs)
            img_grid_mixup = torchvision.utils.make_grid(imgs_mixup)

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

        lr = optimizer.param_groups[-1]['lr']

        # Print results
        print("Epoch {0} Training, Loss {1}, Acc {2}, kaggle Score {3}, lr {4}".format(epoch, 
                                                                                       train_total_err, 
                                                                                       train_total_acc, 
                                                                                       train_kaggle_score,
                                                                                       lr))
        evaluator.clear_cache()

        writer_tensorboard.flush()

        if scheduler is not None:
            if scheduler_type == 'ReduceLROnPlateau':
                scheduler.step(val_total_err)

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
                'lr': lr
            }
        )
        pickle.dump(perf_trace, open(perf_path, 'wb'))

        # store epoch results separately
        epoch_results = {
            'epoch': epoch,
            'train_result': train_result,
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
