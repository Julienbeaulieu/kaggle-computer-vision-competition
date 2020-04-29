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
import pickle
import torch
import click
import numpy as np
from docopt import docopt
#from apex import amp
from src.modeling.meta_arch.build import build_model
from src.data.bengali_data import build_data_loader
from src.modeling.solver import build_optimizer, build_scheduler, build_evaluator, MixupAugmenter

from yacs.config import CfgNode
from src.config import get_cfg_defaults, combine_cfgs

@click.command()
@click.argument('path_output', type=click.Path(exists=True))
@click.option('--path_cfg_data', type=click.Path(exists=True), help="CFG file containing data path node which will be used to overwrite default behaviour.")
@click.option('--path_cfg_override', type=click.Path(exists=True), help="CFG file which will be used to overwrite default and data behaviour.")
def update_cfg_outpath(path_output, cfg):

    cfg = combine_cfgs(cfg)

    cfg.OUTPUT_PATH = path_output

    # Execute training base on the configuration
    return cfg


def train(cfg, debug=False):

    # FILES, PATHS
    assert cfg.OUTPUT_PATH != ''
    output_path = cfg.OUTPUT_PATH
    train_path = cfg.DATASET.TRAIN_DATA_PATH
    val_path = cfg.DATASET.VAL_DATA_PATH
    train_path_sample = cfg.DATASET.TRAIN_DATA_SAMPLE
    valid_path_sample = cfg.DATASET.VALID_DATA_SAMPLE

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    backup_dir = os.path.join(output_path, 'model_backups')
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)

    results_dir = os.path.join(output_path, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    cfg.dump(stream=open(os.path.join(output_path, 'config.yaml'), 'w'))
    state_fpath = os.path.join(output_path, 'model.pt')

    perf_path = os.path.join(results_dir, 'trace.p')
    perf_trace = []

    if debug:
        train_data = pickle.load(open(train_path_sample, 'rb'))
        val_data = pickle.load(open(valid_path_sample, 'rb'))

    # DATA LOADER
    if cfg.DATASET.USE_FOLDS_DATA:
        data_path = cfg.DATASET.FOLDS_PATH
        all_data_folds = pickle.load(open(data_path, 'rb'))
        val_fold = cfg.DATASET.VALIDATION_FOLD
        train_data = []
        val_data = []
        for idx, entries in enumerate(all_data_folds):
            if idx == val_fold:
                val_data = entries
            else:
                train_data = train_data + entries
    else:
        train_data = pickle.load(open(train_path, 'rb'))
        val_data = pickle.load(open(val_path, 'rb'))

    # witchcraft: only train on few classes
    focus_cls = cfg.DATASET.FOCUS_CLASS
    if len(focus_cls) > 0:
        train_data = [x for x in train_data if x[1][0] in focus_cls]
        val_data = [x for x in val_data if x[1][0] in focus_cls]

    train_loader = build_data_loader(train_data, cfg.DATASET, True)
    val_loader = build_data_loader(val_data, cfg.DATASET, False)

    # MODEL
    model = build_model(cfg.MODEL)
    solver_cfg = cfg.MODEL.SOLVER
    total_epochs = solver_cfg.TOTAL_EPOCHS
    loss_fn = solver_cfg.LOSS.NAME
    # for weighted focal loss, initialize last layer bias weights as constant
    if loss_fn == 'weighted_focal_loss':
        last_layer = model.head.fc_layers[-1]
        for m in last_layer.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.bias, -3.0)

    current_epoch = 0
    multi_gpu_training = cfg.MULTI_GPU_TRAINING
    if cfg.RESUME_PATH != "":
        checkpoint = torch.load(cfg.RESUME_PATH, map_location='cpu')
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint["model_state"])
    if multi_gpu_training:
        model = torch.nn.DataParallel(model)
    _ = model.cuda()

    # optimizer, scheduler, amp
    opti_cfg = solver_cfg.OPTIMIZER
    optimizer = build_optimizer(model, opti_cfg)
    use_amp = solver_cfg.AMP

    # ------ Uncomment if we use apex library --------
    # if use_amp:
    #     opt_level = 'O1'
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    if cfg.RESUME_PATH != "":
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        # ------ Uncomment if we use apex library --------
        # if use_amp and 'amp_state' in checkpoint:
        #     amp.load_state_dict(checkpoint['amp_state'])

    scheduler_cfg = solver_cfg.SCHEDULER
    scheduler_type = scheduler_cfg.NAME
    scheduler = build_scheduler(optimizer, scheduler_cfg)

    # evaluator
    mixup_training = solver_cfg.MIXUP_AUGMENT
    if mixup_training:
        mixup_augmenter = MixupAugmenter(solver_cfg.MIXUP)

    evaluator, mixup_evaluator = build_evaluator(solver_cfg)
    evaluator.float().cuda()
    if mixup_evaluator is not None:
        mixup_evaluator.float().cuda()

    s_time = time.time()
    parameters = list(model.parameters())
    for epoch in range(current_epoch, total_epochs):
        model.train()
        if multi_gpu_training:
            model.freeze_bn()
        print('Start epoch', epoch)
        train_itr = iter(train_loader)
        total_err = 0
        total_acc = 0
        inputs, labels = next(train_itr)

        for idx, (inputs, labels) in enumerate(train_itr):

            # compute
            input_data = inputs.float().cuda()
            labels = labels.cuda()
            if mixup_training:
                input_data, labels = mixup_augmenter(input_data, labels)
            grapheme_logits, vowel_logits, consonant_logits = model(input_data)

            if mixup_training:
                eval_result = mixup_evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
            else:
                eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
            optimizer.zero_grad()
            loss = eval_result['loss']


            # ------ Uncomment if we use apex library --------
            # if use_amp:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()


            loss.backward()
            max_grad = torch.max(parameters[-1].grad)
            if not torch.isnan(max_grad):
                optimizer.step()
            else:
                print('NAN in gradient, skip this step')
                optimizer.zero_grad()

            eval_result = {k: eval_result[k].item() for k in eval_result}
            if idx % 100 == 0:
                t_time = time.time()
                print(idx, eval_result['loss'], eval_result['acc'], t_time - s_time)
                s_time = time.time()
        if mixup_training:
            train_result = mixup_evaluator.evalulate_on_cache()
            mixup_evaluator.clear_cache()
        else:
            train_result = evaluator.evalulate_on_cache()
        train_total_err = train_result['loss']
        train_total_acc = train_result['acc']
        train_kaggle_score = train_result['kaggle_score']
        print("Epoch {0} Training, Loss {1}, Acc {2}".format(epoch, train_total_err, train_total_acc))
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

        val_result = evaluator.evalulate_on_cache()
        val_total_err = val_result['loss']
        val_total_acc = val_result['acc']
        val_kaggle_score = val_result['kaggle_score']

        print("Epoch {0} Eval, Loss {1}, Acc {2}".format(epoch, val_total_err, val_total_acc))
        evaluator.clear_cache()

        if scheduler is not None:
            if scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_total_err)
            else:
                scheduler.step()

        print("Saving the model (epoch %d)" % epoch)
        save_state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        if scheduler is not None:
            save_state['scheduler_state'] = scheduler.state_dict()

        # ------ Uncomment if we use apex library --------
        # if use_amp:
        #     save_state['amp_state'] = amp.state_dict()
        torch.save(save_state, state_fpath)

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
                'val_kaggle_score': val_kaggle_score
            }
        )
        pickle.dump(perf_trace, open(perf_path, 'wb'))

        # store epoch full result separately
        epoch_result = {
            'epoch': epoch,
            'train_result': train_result,
            'val_result': val_result
        }
        pickle.dump(epoch_result, open(os.path.join(results_dir, 'result_epoch_{0}.p'.format(epoch)), 'wb'))
        
        # output_path_base = os.path.basename(output_path)
        # os.system('aws s3 sync /root/bengali_data/{0} s3://eaitest1/{1}'.format(output_path_base, output_path_base))
        # os.system('rm -r /root/bengali_data/{0}/model_backups'.format(output_path_base))
        # os.system('mkdir /root/bengali_data/{0}/model_backups'.format(output_path_base))



if __name__ == '__main__':

    # arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    # output_path = arguments['-o']
    # data_path = arguments['--data_cfg']
    # cfg_path = arguments['--cfg']
    # cfg = combine_cfgs(cfg_path)
    # cfg.OUTPUT_PATH = output_path

    update_cfg_outpath()
    #train(cfg, debug=cfg.DEBUG)