# uncomment for py-spy profiler:
# import sys
# sys.path.append('../')
# from src.config.config import cfg

import os
import time
import json
import math
import scipy
from scipy.interpolate import UnivariateSpline
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


def find_lr(cfg, max_iter=1400, init_value=1e-6, final_value=1.0):
    '''
    WIP
    We track the losses given different lr values. 
    Same training loop, but we update the lr according to an update step for each batch iteration
    We apply a smoothing function to the losses for better visualization afterward. 
    '''
    # FILES, PATHS
    train_path = cfg.DATASET.TRAIN_DATA_PATH


    # DATA LOADER
    train_data = pickle.load(open(train_path, 'rb'))
    train_loader = build_data_loader(train_data, cfg.DATASET, True)

    # MODEL
    model = build_model(cfg.MODEL)
    model.cuda()

    # Solver evaluator
    solver_cfg = cfg.MODEL.SOLVER

    total_epochs = solver_cfg.SCHEDULER.TOTAL_EPOCHS


    # Build optimizerW
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

    # find_lr variables
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (2 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss, batch_num = 0.0, 0
    losses, log_lrs = [], []

    model.train()
    train_itr = iter(train_loader)
    for idx, (inputs, labels) in enumerate(train_itr):
        batch_num += 1
        # compute
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
            
        # Stopping condition: if loss explodes ogir idx = 2000
        if batch_num > 1 and loss > 4 * best_loss or idx == max_iter:
            losses = [x.item() for x in losses]
            losses = smoothen_by_spline(log_lrs, losses, s=4)
            return log_lrs[10:-5], losses[10:-5]

        # Record the best loss
        if loss < best_loss or batch_num == 1:
            best_loss = loss
            
        # Store the values 
        losses.append(loss)
        log_lrs.append(math.log10(lr))
        
        # Do the backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eval_result = {k: eval_result[k].item() for k in eval_result}

        if idx % 100 == 0:
            print(idx, eval_result['loss'], eval_result['acc'])
        
        # update the lr
        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr

    #return log_lrs[10:-5], losses[10:-5]


def smoothen_by_spline(xs, ys, **kwargs):
    xs = np.arange(len(ys))
    spl = scipy.interpolate.UnivariateSpline(xs, ys, **kwargs)
    ys = spl(xs)
    return ys

# uncomment for py-spy profiler:
# if __name__=="__main__":
#     log_lrs, losses = find_lr(cfg)