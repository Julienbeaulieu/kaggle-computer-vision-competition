"""

train model
Usage:
    train.py -path_output=<path> [--path_cfg_data=<path>]  [--path_cfg_override=<path>]
    train.py -h | --help

Options:
    -h --help               show this screen help
    -path_output=<path>               output path
    --path_cfg_data=<path>       data config path [default: configs/data.yaml]
    --path_cfg_override=<path>            training config path
"""

# Adapted from Ming's /tools/train.py
import os
import time
import pickle
import torch
import click
#from docopt import docopt
from src.modeling.meta_arch.build import build_model
from src.data.bengali_data import build_data_loader
from src.data.load_datasets import update_cfg_using_dotenv
from src.modeling.solver.optimizer import build_optimizer
from src.modeling.solver.evaluation import build_evaluator
from yacs.config import CfgNode
from src.config import get_cfg_defaults
from datetime import datetime
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# For MLFlow integration
from mlflow import log_metrics, log_param, log_artifact
import mlflow

# For Tensorboard integration
from torch.utils.tensorboard import SummaryWriter

from src.config.config import combine_cfgs

@click.command()
@click.argument('path_output', type=click.Path(exists=True))
@click.option('--path_cfg_data', type=click.Path(exists=True), help="CFG file containing data path node which will be used to overwrite default behaviour.")
@click.option("--path_cfg_override", type=click.Path(exists=True), help="CFG file which will be used to overwrite default and data behaviour.")
def handle_cfg(path_output, data_cfg, cfg):

    cfg = combine_cfgs(data_cfg, cfg)

    cfg.OUTPUT_PATH = path_output

    # Execute training base on the configuration
    train(cfg)



def train(cfg: CfgNode):
    """
    Take a configuration node and train the model from it.
    :param cfg:
    :return:
    """
    # Create writable timestamp for easier record keeping
    timestamp = datetime.now().isoformat(sep="T", timespec="auto")
    name_timestamp = timestamp.replace(":", "_")

    # Start the mlflow run:
    mlflow.start_run(run_name=name_timestamp)

    # Check valid output path, set path from the path_cfg_override modules respectively
    assert cfg.OUTPUT_PATH != ''
    path_output = cfg.OUTPUT_PATH  # output folder
    path_train = cfg.DATASET.TRAIN_DATA_PATH  # training data folder
    path_val = cfg.DATASET.VAL_DATA_PATH  # validation data folder

    # Make output dir and its parents if not exist.
    if not os.path.exists(path_output):
        os.makedirs(path_output)



    # Make result folders if they do not exist.
    results_dir = (Path(path_output) / name_timestamp)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Make backup folders if they do not exist.
    backup_dir = os.path.join(results_dir, 'model_backups')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    writer_tensorboard = SummaryWriter(log_dir=Path(results_dir / "logs_tensorflow"))

    # Now that CFG has been properly merged with new data along the way, time to dump a version of it into a string for trackability purposes.
    cfg.dump(stream=open(os.path.join(results_dir, f'config{name_timestamp}.yaml'), 'w'))

    # file path to store the state of the model.
    state_fpath = os.path.join(results_dir, f'model{name_timestamp}.pt')

    # ????
    perf_path = os.path.join(results_dir, f'trace{name_timestamp}.p')
    perf_trace = []

    # Load data, create the data loader objects from them.
    data_train = pickle.load(open(path_train, 'rb'))
    data_val = pickle.load(open(path_val, 'rb'))
    loader_train = build_data_loader(data_train, cfg.DATASET, True)
    loader_val = build_data_loader(data_val, cfg.DATASET, False)

    # Build the model using configue dict node
    model = build_model(cfg.MODEL)

    # Enable parallel multi GPU mode if the config specify it.
    if cfg.MODEL.PARALLEL:
        print("Utilized parallel processing")
        model = torch.nn.DataParallel(model)

    current_epoch = 0

    # For resuming training (i.e. load checkpoint)
    if cfg.RESUME_PATH != "":
        checkpoint = torch.load(cfg.RESUME_PATH, map_location='cpu')
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint["model_state"])
    _ = model.cuda()

    # SOLVER EVALUATOR
    solver_cfg = cfg.MODEL.SOLVER

    # Build optimizer (between train/validation, using the solver portion of the configuration.
    optimizer = build_optimizer(model, solver_cfg)

    # Build evaluator (between train/validation, using the solver portion of the configuration.
    evaluator = build_evaluator(solver_cfg)

    evaluator.float().cuda()
    total_epochs = solver_cfg.TOTAL_EPOCHS


    # Main training epoch loop starts here.
    for epoch in range(current_epoch, total_epochs):
        # Train an epoch
        model.train()
        print('Start epoch', epoch)
        train_itr = iter(loader_train)
        total_err = 0
        total_acc = 0
        for idx, (data_pixel, data_labels) in enumerate(train_itr):

            # compute
            input_data = data_pixel.float().cuda()
            data_labels = data_labels.cuda()

            # Use the model the produce the classification
            grapheme_logits, vowel_logits, consonant_logits = model(input_data)

            # produce evaluator results
            eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, data_labels)

            # set optimizer to zero.
            optimizer.zero_grad()

            # back propogate the evaluation results.
            eval_result['loss'].backward()

            # optimizer take step forward.
            optimizer.step()

            # tabulate the steps from the evaluation
            eval_result = {k: eval_result[k].item() for k in eval_result}

            # update every hundreds' of
            if idx % 100 == 0:
                print(idx, eval_result['loss'], eval_result['acc'])

        train_result = evaluator.evalulate_on_cache()



        train_total_err = train_result['loss']
        writer_tensorboard.add_scalar('Loss/Train', train_total_err, global_step=epoch)
        #log_metric('loss', train_total_err)

        train_total_acc = train_result['acc']
        writer_tensorboard.add_scalar('Accuracy/Train', train_total_acc, global_step=epoch)
        #log_metric('acc', train_total_acc)

        train_kaggle_score = train_result['kaggle_score']
        writer_tensorboard.add_scalar('Kaggle_Score/Train', train_kaggle_score, global_step=epoch)
        #log_metric('kaggle_score', train_kaggle_score)

        dict_metrics_train = {
            'Loss/Train': train_total_err,
            'Accuracy/Train': train_total_acc,
            'Kaggle_Score/Train': train_kaggle_score,
        }
        log_metrics(dict_metrics_train, step=epoch)

        print(f"Epoch {epoch} Training, Loss {train_total_err}, Acc {train_total_acc}")
        evaluator.clear_cache()

        # compute validation error
        model.eval()
        val_itr = iter(loader_val)
        with torch.no_grad():
            for idx, (data_pixel, data_labels) in enumerate(val_itr):
                input_data = data_pixel.float().cuda()
                data_labels = data_labels.cuda()
                grapheme_logits, vowel_logits, consonant_logits = model(input_data)
                eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, data_labels)
                eval_result = {k: eval_result[k].item() for k in eval_result}
                total_err += eval_result['loss']
                total_acc += eval_result['acc']
                # print(total_err / (1 + idx), total_acc / (1 + idx))

        val_result = evaluator.evalulate_on_cache()

        val_total_err = val_result['loss']
        writer_tensorboard.add_scalar('Loss/Val', val_total_err, global_step=epoch)

        val_total_acc = val_result['acc']
        writer_tensorboard.add_scalar('Accuracy/Val', val_total_acc, global_step=epoch)

        val_kaggle_score = val_result['kaggle_score']
        writer_tensorboard.add_scalar('Kaggle_Score/Val', val_kaggle_score, global_step=epoch)

        dict_metrics_val = {
            'Loss/Validation': val_total_err,
            'Accuracy/Validation': val_total_acc,
            'Kaggle_Score/Validation': val_kaggle_score,
        }
        log_metrics(dict_metrics_val, step=epoch)

        # Write to disk.
        writer_tensorboard.flush()

        print(f"Epoch {epoch} Eval, Loss {val_total_err}, Acc {val_total_acc}")
        evaluator.clear_cache()

        print("Saving the model (epoch %d)" % epoch)
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, state_fpath)

        print(f"Making a backup (step {epoch})")
        backup_fpath = os.path.join(backup_dir, f"model_bak_{epoch}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, backup_fpath)

        # Dump the traces
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

    mlflow.end_run()
if __name__ == '__main__':

    # Obtain some key arguments with regard to the path of output, data, path_cfg_override files.
    handle_cfg()

