# uncomment for py-spy profiler:
# import sys
# sys.path.append('../')
# from src.config.config import cfg

import os
import json
import pickle
import torch
import math
import numpy as np
import scipy

from scipy.interpolate import UnivariateSpline
from src.modeling.meta_arch.build import build_model
from src.data.bengali_data import build_data_loader
from src.modeling.solver.optimizer import build_optimizer
from src.modeling.solver.evaluation import build_evaluator


def find_lr(cfg, max_iter=400, init_value=1e-6, final_value=1.0):
    '''
    WIP
    We track the losses given different lr values. 
    Same training loop, but we update the lr according to an update step for each batch iteration
    We apply a smoothing function to the losses for better visualization afterward. 
    '''
    # FILES, PATHS
    train_path = cfg.DATASET.TRAIN_DATA_PATH
    val_path = cfg.DATASET.VAL_DATA_PATH 

    # DATA LOADER
    train_data = pickle.load(open(train_path, 'rb'))
    val_data = pickle.load(open(val_path, 'rb'))
    train_loader = build_data_loader(train_data, cfg.DATASET, True)
    val_loader = build_data_loader(val_data, cfg.DATASET, False)

    # MODEL
    model = build_model(cfg.MODEL)
    current_epoch = 0
    if cfg.RESUME_PATH != "":
        checkpoint = torch.load(cfg.RESUME_PATH, map_location='cpu')
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint["model_state"])
    _ = model.cuda()

    # SOLVER EVALUATOR
    solver_cfg = cfg.MODEL.SOLVER
    optimizer = build_optimizer(model, solver_cfg)
    evaluator = build_evaluator(solver_cfg)
    evaluator.float().cuda()
    total_epochs = solver_cfg.TOTAL_EPOCHS

    # find_lr variables
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (2 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss, batch_num = 0.0, 0
    losses, log_lrs = [], []

    model.train()
    train_itr = iter(train_loader)
    total_err = 0
    total_acc = 0
    for idx, (inputs, labels) in enumerate(train_itr):
        batch_num += 1
        # compute
        input_data = inputs.float().cuda()
        labels = labels.cuda()
        grapheme_logits, vowel_logits, consonant_logits = model(input_data)

        eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
        
        # keep track of the loss
        loss = eval_result['loss']
            
        # Stopping condition: if loss explodes ogir idx = 2000
        if batch_num > 1 and loss > 4 * best_loss or idx == 2000:
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
        eval_result['loss'].backward()
        optimizer.step()

        eval_result = {k: eval_result[k].item() for k in eval_result}
        total_err += eval_result['loss']
        total_acc += eval_result['acc']
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