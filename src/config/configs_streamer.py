from pathlib import Path
import os
import subprocess
import sys
from src.config.config import combine_cfgs
from src.models.model_train import train
import torch
import click
import glob

@click.command()
@click.argument('path_YAMLS', type=click.Path(exists=True))
def run_experiments(path_YAMLS: Path = Path(r"C:\Git\bengali.ai\configs\inbox"),
                    path_Results: Path = Path(r"C:\Git\bengali.ai\results")):
    """
    This monitor the path folder which contains various YAMLs ans will run them consecutively.
    You can use ENVIRONMENTAL variables to specify the hardware ID to be used.
    :param path_YAMLS:
    :return:
    """
    filter_yaml = "*.yaml"

    # Get the inital file list.
    os.chdir(path=path_YAMLS)
    files = glob.glob(filter_yaml)

    while files is not []:
        # Get the updated file lists
        os.chdir(path=path_YAMLS)

        # Monitor YAML folder for YAML files.
        files = glob.glob(filter_yaml)

        # Early exit condition when there are no more YAML files to prevent files[0] error.
        if files == []:
            break

        # Combine from the FIRST experimental file.
        cfg = combine_cfgs(path_cfg_override=Path(path_YAMLS / files[0]))

        # Use the Output Path.
        cfg.OUTPUT_PATH = path_Results
        print(f"Conducting Experiment {files[0]}: {cfg.DESCRIPTION} on Device{torch.cuda.current_device()}:{torch.cuda.get_device_name(torch.cuda.current_device())}")

        train(cfg)

        # Remove the file to eventually empty the directory.
        # This is ONLY done at the end when the training is SUCCESFUL
        # When that happens, the yaml file will be saved already in the result folder and no longer needed at the source locations.
        #
        os.remove(files[0])

if __name__=="__main__":
    run_experiments()