from pathlib import Path
import os
import subprocess
import sys
from src.config.config import combine_cfgs
from src.models.model_train import train
import torch
import click

@click.command()
@click.argument('path_YAMLS', type=click.Path(exists=True))
def run_experiments(path_YAMLS: Path = Path(r"C:\Git\bengali.ai\configs\inbox")):
    """
    This monitor the path folder which contains various YAMLs ans will run them consecutively.
    You can use ENVIRONMENTAL variables to specify the hardware ID to be used.
    :param path_YAMLS:
    :return:
    """

    # Monitor YAML folder.
    files = os.listdir(path=path_YAMLS)

    # Merge YAML iteratively.
    for file in files:
        cfg = combine_cfgs(path_cfg_override=Path(path_YAMLS / file))
        cfg.OUTPUT_PATH = r"C:\Git\bengali.ai\results"
        print(f"Conducting Expeirment {file}: {cfg.DESCRIPTION} on {torch.cuda.get_device_name(torch.cuda.current_device())}")
        train(cfg)
        #os.remove(file)

if __name__=="__main__":
    run_experiments()