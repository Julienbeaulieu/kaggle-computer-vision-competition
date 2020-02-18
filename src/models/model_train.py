"""

train model
Usage:
    train.py -path_output=<path> [--data_cfg=<path>]  [--cfg=<path>]
    train.py -h | --help

Options:
    -h --help               show this screen help
    -path_output=<path>               output path
    --data_cfg=<path>       data config path [default: configs/data.yaml]
    --cfg=<path>            training config path
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
from mlflow import log_metric, log_param, log_artifact


# For Tensorboard integration
from torch.utils.tensorboard import SummaryWriter


@click.command()
@click.argument('path_output', type=click.Path(exists=False))
@click.option('--data_cfg', type=click.Path(exists=True), help="CFG file containing data path node which will be used to overwrite default behaviour.")
@click.option("--cfg", type=click.Path(exists=True), help="CFG file which will be used to overwrite default and data behaviour.")
def handle_cfg(path_output, data_cfg, cfg):

    # path to output files
    output_path = path_output

    # path to cfg_data files
    data_path = data_cfg

    # path to cfg actual
    cfg_path = cfg

    # Path order of precedence is:
    # Priority 1, 2, 3, 4 respectively
    # .env > other CFG YAML > data.yaml > default.yaml

    # Load default lowest tier one:
    # Priority 4:
    cfg = get_cfg_defaults()

    # Merge from the path_data
    # Priority 3:
    if data_path is not None and os.path.exists(data_path):
        cfg.merge_from_file(data_path)

    # Merge from other cfg_path files to further reduce effort
    # Priority 2:
    if cfg_path is not None and os.path.exists(cfg_path):
        cfg.merge_from_file(cfg_path)

    # Merge from .env
    # Priority 1:
    list_cfg = update_cfg_using_dotenv()
    if list_cfg is not []:
        cfg.merge_from_list(list_cfg)

    #
    cfg.OUTPUT_PATH = output_path

    # Execute training base on the configuration
    train(cfg)



def train(cfg: CfgNode):
    """
    Take a configuration node and train the model from it.
    :param cfg:
    :return:
    """
    # Check valid output path, set path from the cfg modules respectively
    assert cfg.OUTPUT_PATH != ''
    path_output = cfg.OUTPUT_PATH  # output folder
    path_train = cfg.DATASET.TRAIN_DATA_PATH  # training data folder
    path_val = cfg.DATASET.VAL_DATA_PATH  # validation data folder

    # Make output dir and its parents if not exist.
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # Create writable timestamp for easier record keeping
    timestamp = datetime.now().isoformat(sep="T", timespec="auto")
    name_timestamp = timestamp.replace(":", "_")

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
        writer_tensorboard.add_scalar('loss', train_total_err)
        log_metric('loss', train_total_err)

        train_total_acc = train_result['acc']
        writer_tensorboard.add_scalar('acc', train_total_acc)
        log_metric('acc', train_total_acc)

        train_kaggle_score = train_result['kaggle_score']
        writer_tensorboard.add_scalar('kaggle_score', train_kaggle_score)
        log_metric('kaggle_score', train_kaggle_score)

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
        writer_tensorboard.add_scalar('val_loss', val_total_err)
        log_metric('val_loss', val_total_err)

        val_total_acc = val_result['acc']
        writer_tensorboard.add_scalar('val_total_acc', val_total_acc)
        log_metric('val_acc', val_total_acc)

        val_kaggle_score = val_result['kaggle_score']
        writer_tensorboard.add_scalar('val_kaggle_score', val_kaggle_score)
        log_metric('val_kaggle_score', val_kaggle_score)


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


if __name__ == '__main__':

    #arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)

    # Obtain some key arguments with regard to the path of output, data, cfg files.
    handle_cfg()

