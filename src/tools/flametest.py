import os
import json
import pickle
import torch
import sys
sys.path.append('../')
from src.modeling.meta_arch.build import build_model
from src.data.bengali_data import build_data_loader
from src.modeling.solver.optimizer import build_optimizer
from src.modeling.solver.evaluation import build_evaluator
from src.config.config import cfg



# FILES, PATHS
assert cfg.OUTPUT_PATH != ''
output_path = cfg.OUTPUT_PATH
train_path = cfg.DATASET.TRAIN_DATA_0

# DATA LOADER
train_data = pickle.load(open(train_path, 'rb'))
train_loader = build_data_loader(train_data, cfg.DATASET, True)

# MODEL
model = build_model(cfg.MODEL)

# SOLVER EVALUATOR
solver_cfg = cfg.MODEL.SOLVER
optimizer = build_optimizer(model, solver_cfg)
evaluator = build_evaluator(solver_cfg)
evaluator.float().cuda()
total_epochs = solver_cfg.TOTAL_EPOCHS


model.train()
print('Start training')
train_itr = iter(train_loader)
total_err = 0
total_acc = 0
for idx, (inputs, labels) in enumerate(train_itr):

    # compute
    input_data = inputs.float().cuda()
    labels = labels.cuda()
    grapheme_logits, vowel_logits, consonant_logits = model(input_data)

    eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
    optimizer.zero_grad()
    eval_result['loss'].backward()
    optimizer.step()

    eval_result = {k: eval_result[k].item() for k in eval_result}
    total_err += eval_result['loss']
    total_acc += eval_result['acc']
    if idx % 100 == 0:
        print(idx, eval_result['loss'], eval_result['acc'])
    if idx == 150: break

train_total_err = total_err / (1 + idx)
train_total_acc = total_acc / (1 + idx)
print("Epoch {0} Training, Loss {1}, Acc {2}".format(epoch, train_total_err, train_total_acc))

        
