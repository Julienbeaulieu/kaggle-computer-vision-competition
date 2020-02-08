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
import pickle
import torch
from docopt import docopt
from src.modeling.meta_arch.build import build_model
from src.data.bengali_data import build_data_loader
from src.modeling.solver.optimizer import build_optimizer
from src.modeling.solver.evaluation import build_evaluator
from yacs.config import CfgNode
from src.config import get_cfg_defaults


def train(cfg: CfgNode):
    # FILES, PATHS
    assert cfg.OUTPUT_PATH != ''
    output_path = cfg.OUTPUT_PATH
    train_path = cfg.DATASET.TRAIN_DATA_PATH
    val_path = cfg.DATASET.VAL_DATA_PATH

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

    for epoch in range(current_epoch, total_epochs):
        model.train()
        print('Start epoch', epoch)
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
            if idx % 100 == 0:
                print(idx, eval_result['loss'], eval_result['acc'])

        train_result = evaluator.evalulate_on_cache()
        train_total_err = train_result['loss']
        train_total_acc = train_result['acc']
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
                # print(total_err / (1 + idx), total_acc / (1 + idx))

        val_result = evaluator.evalulate_on_cache()
        val_total_err = val_result['loss']
        val_total_acc = val_result['acc']
        print("Epoch {0} Eval, Loss {1}, Acc {2}".format(epoch, val_total_err, val_total_acc))
        evaluator.clear_cache()

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
                'val_err': val_total_err,
                'val_acc': val_total_acc
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


if __name__ == '__main__':

    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    output_path = arguments['-o']
    data_path = arguments['--data_cfg']
    cfg_path = arguments['--cfg']
    cfg = get_cfg_defaults()
    cfg.merge_from_file(data_path)
    if cfg_path is not None:
        cfg.merge_from_file(cfg_path)
    cfg.OUTPUT_PATH = output_path
    train(cfg)