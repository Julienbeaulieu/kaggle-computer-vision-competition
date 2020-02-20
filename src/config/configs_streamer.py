from pathlib import Path
import os
import subprocess
import sys
from src.config.config import combine_cfgs
from src.models.model_train import train

def run_experiments(path_YAMLS: Path = Path(r"C:\Git\bengali.ai\configs\inbox")):

    # Monitor YAML folder.
    files = os.listdir(path=path_YAMLS)

    # Merge YAML iteratively.
    for file in files:
        cfg = combine_cfgs(path_cfg_override=Path(path_YAMLS / file))
        cfg.OUTPUT_PATH = r"C:\Git\bengali.ai\results"
        print(f"Conducting Expeirment {file}: {cfg.DESCRIPTION}")
        train(cfg)
        os.remove(file)

if __name__=="__main__":
    run_experiments()