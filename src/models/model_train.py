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
from src.modeling.solver.optimizer import build_optimizer
from src.modeling.solver.evaluation import build_evaluator
from yacs.config import CfgNode as ConfigurationNode
from src.config import get_cfg_defaults
from datetime import datetime
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# For MLFlow integration
from mlflow import log_metrics, log_param, log_artifact
import mlflow

# For Tensorboard integration
from torch.utils.tensorboard import SummaryWriter

from src.config.config import combine_cfgs, update_cfg_using_dotenv


@click.command()
@click.argument('path_output', type=click.Path(exists=True))
@click.option('--path_cfg_data', type=click.Path(exists=True), help="CFG file containing data path node which will be used to overwrite default behaviour.")
@click.option("--path_cfg_override", type=click.Path(exists=True), help="CFG file which will be used to overwrite default and data behaviour.")
def update_cfg_outpath(path_output, data_cfg, cfg):

    cfg = combine_cfgs(data_cfg, cfg)

    cfg.OUTPUT_PATH = path_output

    # Execute training base on the configuration
    return cfg

class TrainingMixin:
    def train(self, config: ConfigurationNode = None):
        """
        Take a configuration node and train the model from it.
        :param config:
        :return:
        """
        if config is None:
            config = self.config
        # Create writable timestamp for easier record keeping
        timestamp = datetime.now().isoformat(sep="T", timespec="auto")
        name_timestamp = timestamp.replace(":", "_")

        # Start the mlflow run:
        mlflow.start_run(run_name=name_timestamp)

        # Check valid output path, set path from the path_cfg_override modules respectively
        assert config.OUTPUT_PATH != ''
        path_output = config.OUTPUT_PATH  # output folder
        path_train = config.DATASET.TRAIN_DATA_PATH  # training data folder
        path_val = config.DATASET.VAL_DATA_PATH  # validation data folder

        # Make output dir and its parents if not exist.
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        # Make result folders if they do not exist.
        self.results_dir = (Path(path_output) / name_timestamp)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Make backup folders if they do not exist.
        self.backup_dir = os.path.join(self.results_dir, 'model_backups')
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

        writer_tensorboard = SummaryWriter(log_dir=Path(self.results_dir / "logs_tensorflow"))

        # Now that CFG has been properly merged with new data along the way, time to dump a version of it into a string for trackability purposes.
        config.dump(stream=open(os.path.join(self.results_dir, f'config{name_timestamp}.yaml'), 'w'))

        # file path to store the state of the model.
        state_fpath = os.path.join(self.results_dir, f'model{name_timestamp}.pt')

        # ????
        perf_path = os.path.join(self.results_dir, f'trace{name_timestamp}.p')
        perf_trace = []

        # Load data, create the data loader objects from them.
        data_train = pickle.load(open(path_train, 'rb'))
        data_val = pickle.load(open(path_val, 'rb'))
        self.loader_train = build_data_loader(data_train, config.DATASET, True)
        self.loader_val = build_data_loader(data_val, config.DATASET, False)

        # Build the model using configue dict node
        self.model = build_model(config.MODEL)

        # Enable parallel multi GPU mode if the config specify it.
        if config.MODEL.PARALLEL:
            print("Utilized parallel processing")
            self.model = torch.nn.DataParallel(self.model)

        current_epoch = 0

        # For resuming training (i.e. load checkpoint)
        if config.RESUME_PATH != "":
            checkpoint = torch.load(config.RESUME_PATH, map_location='cpu')
            current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint["model_state"])
        _ = self.model.cuda()

        # SOLVER EVALUATOR
        cfg_solver = config.MODEL.SOLVER

        # Build optimizer (between train/validation, using the solver portion of the configuration.
        optimizer = build_optimizer(self.model, cfg_solver)

        # Build evaluator (between train/validation, using the solver portion of the configuration.
        evaluator = build_evaluator(cfg_solver)

        evaluator.float().cuda()
        total_epochs = cfg_solver.TOTAL_EPOCHS


        # Main training epoch loop starts here.
        for epoch in range(current_epoch, total_epochs):

            # Train a single epoch
            self.train_epoch(epoch, evaluator, optimizer, perf_path, perf_trace, state_fpath, writer_tensorboard)

        mlflow.end_run()

    def train_epoch(self, epoch, evaluator, optimizer, perf_path, perf_trace, state_fpath, writer_tensorboard):
        """
        Trains an epoch, output it to the right place.
        :param epoch:
        :param evaluator:
        :param optimizer:
        :param perf_path:
        :param perf_trace:
        :param state_fpath:
        :param writer_tensorboard:
        :return:
        """

        # Train an epoch
        self.model.train()
        print('Start epoch', epoch)
        train_itr = iter(self.loader_train)
        total_err = 0
        total_acc = 0

        for index, (data_pixel, data_labels) in enumerate(train_itr):

            # compute
            input_data = data_pixel.float().cuda()
            data_labels = data_labels.cuda()

            # Use the model the produce the classification
            grapheme_logits, vowel_logits, consonant_logits = self.model(input_data)

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
            if index % 100 == 0:
                print(index, eval_result['loss'], eval_result['acc'])
        train_result = evaluator.evalulate_on_cache()
        train_total_err = train_result['loss']
        writer_tensorboard.add_scalar('Loss/Train', train_total_err, global_step=epoch)
        # log_metric('loss', train_total_err)
        train_total_acc = train_result['acc']
        writer_tensorboard.add_scalar('Accuracy/Train', train_total_acc, global_step=epoch)
        # log_metric('acc', train_total_acc)
        train_kaggle_score = train_result['kaggle_score']
        writer_tensorboard.add_scalar('Kaggle_Score/Train', train_kaggle_score, global_step=epoch)
        # log_metric('kaggle_score', train_kaggle_score)
        dict_metrics_train = {
            'Loss/Train': train_total_err,
            'Accuracy/Train': train_total_acc,
            'Kaggle_Score/Train': train_kaggle_score,
        }
        log_metrics(dict_metrics_train, step=epoch)
        print(f"Epoch {epoch} Training, Loss {train_total_err}, Acc {train_total_acc}")
        evaluator.clear_cache()
        # compute validation error
        self.model.eval()
        val_itr = iter(self.loader_val)
        with torch.no_grad():
            for index, (data_pixel, data_labels) in enumerate(val_itr):
                input_data = data_pixel.float().cuda()
                data_labels = data_labels.cuda()
                grapheme_logits, vowel_logits, consonant_logits = self.model(input_data)
                eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, data_labels)
                eval_result = {k: eval_result[k].item() for k in eval_result}
                total_err += eval_result['loss']
                total_acc += eval_result['acc']
                # print(total_err / (1 + input_index), total_acc / (1 + input_index))
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
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, state_fpath)
        print(f"Making a backup (step {epoch})")
        backup_fpath = os.path.join(self.backup_dir, f"model_bak_{epoch}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state": self.model.state_dict(),
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
        pickle.dump(epoch_result, open(os.path.join(self.results_dir, 'result_epoch_{0}.p'.format(epoch)), 'wb'))


if __name__ == '__main__':

    # Obtain some key arguments with regard to the path of output, data, path_cfg_override files.
    update_cfg_outpath()

