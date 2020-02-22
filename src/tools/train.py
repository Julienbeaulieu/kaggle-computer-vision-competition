import os
import json
import pickle
import torch
from src.modeling.meta_arch.build import build_model
from src.data.bengali_data import build_data_loader
from src.modeling.solver.optimizer import build_optimizer, build_scheduler
from src.modeling.solver.evaluation import build_evaluator

from functools import partial
import math 
from typing import List, Dict, Union, Optional, Iterable

# def is_listy(x): return isinstance(x, (list,tuple))

# def tensor(x, *rest):
#     "Like `torch.as_tensor`, but handle lists too, and can pass multiple vector elements directly."
#     if len(rest): x = (x,)+rest
#     # XXX: Pytorch bug in dataloader using num_workers>0; TODO: create repro and report
#     if is_listy(x) and len(x)==0: return tensor(0)
#     res = torch.tensor(x) if is_listy(x) else torch.as_tensor(x)
#     if res.dtype is torch.int32:
#         #warn('Tensor is int32: upgrading to int64; for better performance use int64 input')
#         return res.long()
#     return res

# class Recorder():
#     def begin_fit(self): self.lrs,self.losses = [],[]

#     def after_batch(self):
#         if not self.in_train: return
#         self.lrs.append(self.opt.param_groups[-1]['lr'])
#         self.losses.append(self.loss.detach().cpu())        

#     def plot_lr  (self): plt.plot(self.lrs)
#     def plot_loss(self): plt.plot(self.losses)

# class ParamScheduler():
#     def __init__(self, param_name, sched_func, opt, n_epochs, epochs):
#         self.param_name = param_name
#         self.sched_func = sched_func
#         self.n_epochs = n_epochs
#         self.epochs = epochs
#         self.opt = opt
    
#     def set_param(self):
#         for pg in self.opt.param_groups:
#             pg[self.param_name] = self.sched_func(self.n_epochs/self.epochs)

#     def begin_batch(self):
#         self.set_param()

# def annealer(f):
#     def _inner(start, end): return partial(f, start, end)
#     return _inner

# @annealer
# def sched_lin(start, end, pos): return start + pos*(end-start)

# @annealer
# def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
# @annealer
# def sched_no(start, end, pos):  return start
# @annealer
# def sched_exp(start, end, pos): return start * (end/start) ** pos

# def cos_1cycle_anneal(start, high, end):
#     return [sched_cos(start, high), sched_cos(high, end)]


# def listify(p=None, q=None):
#     "Make `p` listy and the same length as `q`."
#     if p is None: p=[]
#     elif isinstance(p, str):          p = [p]
#     elif not isinstance(p, Iterable): p = [p]
#     #Rank 0 tensors in PyTorch are Iterable but don't have a length.
#     else:
#         try: a = len(p)
#         except: p = [p]
#     n = q if type(q)==int else len(p) if q is None else len(q)
#     if len(p)==1: p = p * n
#     assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
#     return list(p)

# def combine_scheds(pcts, scheds):
#     assert sum(pcts) == 1.
#     pcts = tensor([0] + listify(pcts))
#     assert torch.all(pcts >= 0)
#     pcts = torch.cumsum(pcts, 0)
#     def _inner(pos):
#         idx = (pos >= pcts).nonzero().max()
#         actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
#         return scheds[idx](actual_pos)
#     return _inner

# #sched = combine_scheds([0.3, 0.7], [sched_cos(0.003, 0.006), sched_cos(0.006, 0.002)]) 
# sched = sched_lin(0.003, 0.006)


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
    state_fpath = os.path.join(output_path, 'model.pt')
    perf_path = os.path.join(output_path, 'trace.p')
    perf_trace = []

    # DATA LOADER
    if debug:
        train_data = pickle.load(open(train_path_sample, 'rb'))
        val_data = pickle.load(open(valid_path_sample, 'rb'))
    else:
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
    scheduler = build_scheduler(optimizer, solver_cfg)
    # scheduler = MultiStepLR(optimizer, milestones=[1,2,3,9], gamma=0.1, last_epoch=-1)
    evaluator = build_evaluator(solver_cfg)
    evaluator.float().cuda()
    total_epochs = solver_cfg.TOTAL_EPOCHS

    for epoch in range(current_epoch, total_epochs):
        scheduler_lrs,scheduler_losses = [],[]
        model.train()
        print('Start epoch', epoch)
        train_itr = iter(train_loader)
        total_err = 0
        total_acc = 0
        inputs, labels = next(train_itr)

        for idx, (inputs, labels) in enumerate(train_itr):
            
            # scheduler = ParamScheduler('lr', sched, optimizer, solver_cfg.TOTAL_EPOCHS, 4000.0)
            # scheduler.set_param()

            # compute
            input_data = inputs.float().cuda()
            labels = labels.cuda()

            # Calculate preds
            grapheme_logits, vowel_logits, consonant_logits = model(input_data)

            # Calling MultiHeadsEval forward function
            eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
            optimizer.zero_grad()

            eval_result['loss'].backward()
            optimizer.step()

            eval_result = {k: eval_result[k].item() for k in eval_result}
            
            # lr sched
            scheduler_lrs.append(optimizer.param_groups[-1]['lr'])
            scheduler_losses.append(eval_result['loss'])
            

            if idx % 100 == 0:
                print(idx, eval_result['loss'], eval_result['acc'])
                
        train_result = evaluator.evalulate_on_cache()
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

        val_result = evaluator.evalulate_on_cache()
        val_total_err = val_result['loss']
        val_total_acc = val_result['acc']
        val_kaggle_score = val_result['kaggle_score']

        print("Epoch {0} Eval, Loss {1}, Acc {2}, Kaggle score {3}".format(epoch, val_total_err, val_total_acc, val_kaggle_score))
        evaluator.clear_cache()

        scheduler.step()
        print('Learning rate now set to: ' + str(optimizer.param_groups[-1]['lr']))

        print("Saving the model (epoch %d)" % epoch)
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, state_fpath)

        print("Making a backup (step %d)" % epoch)
        backup_fpath = os.path.join(backup_dir, "model_bak_%06d.pt" % (epoch,))
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, backup_fpath)

        perf_trace.append(
            {
                'epoch': epoch,
                'train_err': train_total_err,
                'train_acc': train_total_acc,
                'train_kaggle_score': train_kaggle_score,
                'val_err': val_total_err,
                'val_acc': val_total_acc,
                'val_kaggle_score': val_kaggle_score,
                'train_result': train_result,
                'val_result': val_result
            }
        )
        pickle.dump(perf_trace, open(perf_path, 'wb'))
