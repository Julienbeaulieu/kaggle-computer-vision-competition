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
import json
import pickle
import torch
import numpy as np
from docopt import docopt
from src.modeling.meta_arch.build import build_model
from src.data.bengali_data import build_data_loader
from src.modeling.solver import build_optimizer, build_scheduler, build_evaluator, MixupAugmenter

from yacs.config import CfgNode
from src.config import get_cfg_defaults

def train(cfg, debug=False):

    # PATHS
    assert cfg.OUTPUT_PATH != ''
    output_path = cfg.OUTPUT_PATH
    train_path = cfg.DATASET.TRAIN_DATA_PATH
    val_path = cfg.DATASET.VAL_DATA_PATH

    # sample is 1/4th of the train images - aka 1 .parquet file
    train_path_sample = cfg.DATASET.TRAIN_DATA_SAMPLE
    valid_path_sample = cfg.DATASET.VALID_DATA_SAMPLE

    # model backups
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    backup_dir = os.path.join(output_path, 'model_backups')
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)
    
    # results
    results_dir = os.path.join(output_path, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    cfg.dump(stream=open(os.path.join(output_path, 'config.yaml'), 'w'))
    state_fpath = os.path.join(output_path, 'model.pt')

    perf_path = os.path.join(output_path, 'trace.p')
    perf_trace = []

    # debug: load reduced training file
    if debug:
        train_data = pickle.load(open(train_path_sample, 'rb'))
        val_data = pickle.load(open(valid_path_sample, 'rb'))
    else:
        train_data = pickle.load(open(train_path, 'rb'))
        val_data = pickle.load(open(val_path, 'rb'))

    # DataLoader
    train_loader = build_data_loader(train_data, cfg.DATASET, True)
    val_loader = build_data_loader(val_data, cfg.DATASET, False)

    # MODEL
    model = build_model(cfg.MODEL)
    solver_cfg = cfg.MODEL.SOLVER
    loss_fn = solver_cfg.LOSS.NAME

    current_epoch = 0
    total_epochs = solver_cfg.SCHEDULER.TOTAL_EPOCHS

    # resume
    if cfg.RESUME_PATH != "":
        checkpoint = torch.load(cfg.RESUME_PATH, map_location='cpu')
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint["model_state"])
    _ = model.cuda()

    # optimizer, scheduler
    opti_cfg = solver_cfg.OPTIMIZER
    optimizer = build_optimizer(model, opti_cfg)
    sched_cfg = solver_cfg.SCHEDULER
    scheduler = build_scheduler(optimizer, sched_cfg, steps_per_epoch=np.int(len(train_data)/cfg.DATASET.BATCH_SIZE))

    # evaluator
    mixup_training = solver_cfg.MIXUP_AUGMENT
    if mixup_training:
        mixup_augmenter = MixupAugmenter(solver_cfg.MIXUP)
    evaluator, mixup_evaluator = build_evaluator(solver_cfg)
    evaluator.float().cuda()
    if mixup_evaluator is not None:
        mixup_evaluator.float().cuda()

    for epoch in range(current_epoch, total_epochs):
        model.train()
        print('Start epoch', epoch)
        train_itr = iter(train_loader)
        total_err = 0
        total_acc = 0
        inputs, labels = next(train_itr)

        for idx, (inputs, labels) in enumerate(train_itr):
            # compute
            input_data = inputs.float().cuda()
            labels = labels.cuda()

            # Calculate preds
            if mixup_training:
                input_data, labels = mixup_augmenter(input_data, labels)
            grapheme_logits, vowel_logits, consonant_logits = model(input_data)

            # Calling MultiHeadsEval forward function
            if mixup_training:
                eval_result = mixup_evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
            else:
                eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
            optimizer.zero_grad()

            loss = eval_result['loss']
            loss.backward()
            optimizer.step()

            eval_result = {k: eval_result[k].item() for k in eval_result}        
            scheduler.step()
        
            if idx % 100 == 0:
                print(idx, eval_result['loss'], eval_result['acc'])

        if mixup_training:
            train_result = mixup_evaluator.evaluate_on_cache()
            mixup_evaluator.clear_cache()
        else:
            train_result = evaluator.evaluate_on_cache()

        train_total_err = train_result['loss']
        train_total_acc = train_result['acc']
        train_kaggle_score = train_result['kaggle_score']

        print("Epoch {0} Training, Loss {1}, Acc {2}, kaggle Score {3}".format(epoch, train_total_err, train_total_acc, train_kaggle_score))
        evaluator.clear_cache()

        # compute validation error
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

        val_result = evaluator.evaluate_on_cache()
        val_total_err = val_result['loss']
        val_total_acc = val_result['acc']
        val_kaggle_score = val_result['kaggle_score']

        print("Epoch {0} Eval, Loss {1}, Acc {2}, Kaggle score {3}".format(epoch, val_total_err, val_total_acc, val_kaggle_score))
        evaluator.clear_cache()

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

        perf_trace.append(
            {
                'epoch': epoch,
                'train_err': train_total_err,
                'train_acc': train_total_acc,
                'train_kaggle_score': train_kaggle_score,
                'val_err': val_total_err,
                'val_acc': val_total_acc,
                'val_kaggle_score': val_kaggle_score,
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

if __name__ == '__main__':

    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    output_path = arguments['-o']
    data_path = arguments['--data_cfg']
    cfg_path = arguments['--cfg']
    # output_path = cfg.OUTPUT_PATH
    # cfg_path = Path('C:/Users/nasty/data-science/kaggle/bengali-git/bengali.ai/src/config/mobilenet_V2/exp_01.yaml')
    cfg = get_cfg_defaults()
    print(cfg_path)
    cfg.merge_from_file(cfg_path)
    if cfg_path is not None:
        cfg.merge_from_file(cfg_path)
    cfg.OUTPUT_PATH = output_path
    train(cfg)
