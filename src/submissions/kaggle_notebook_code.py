import numpy as np
import pandas as pd
import os, sys
import glob
from pathlib import Path
from shutil import copytree
import pickle

# Default Path
path_input = Path("/kaggle/input/")
path_output = Path("/kaggle/working/")

# Source Data:
path_data = path_input / "bengaliai-cv19"

# Source Data Packages
path_source_code = path_input / "bengaliaipipeline01"
path_target_code = path_output / "bengaliai"

# Source Dependency Packages
path_source_dep = path_input / "bengaliai-pip-dep"
path_target_dep = path_output / "bengaliai_dep"

# =================
# Data Preparation
# =================


#######################################
# Copy from INPUT DATSET to WORKING DIR
# Fom src.submissions.bengaliai import kaggle_project_submission
# This does NOT include the /depends folder.
if not path_target_code.exists():
	copytree(src=path_source_code, dst=path_target_code)


# =======================
# Dependency Preparation
# =======================


class SimpleObj:
	"""
	This class data structure is required to decompress the entry and bin data.
	"""

	def __init__(self, fname, bdata):
		self.name = fname
		self.bdata = bdata
		return


# Copy the binary dependencies generated by the local /depends folder.
if not path_target_dep.exists():
	copytree(src=path_source_dep, dst=path_target_dep)

# print(os.getcwd())

## Remove files:
# shutil.rmtree("/kaggle/working/bengaliai/")

try:
	import yacs
except ImportError as e:
	os.chdir(path_target_dep)
	# The file that contain all the binary of all possible dependencies.
	picklefile = 'dill.pkl'

	# unpack pickle
	installers = []
	with open(picklefile, 'rb') as pf:
		installers = pickle.load(pf)

	for i in installers:
		with open(i.name, 'wb') as p:
			p.write(i.bdata)

	# install
	os.system('pip install -r requirements.txt --no-index --find-links .')

# =======================
# Running Preidction
# =======================
os.chdir(path_target_code)
print(os.getcwd())

path_cfg = "/kaggle/working/bengaliai/results/config2020-02-23T09_33_56.405782.yaml"
path_weight = "/kaggle/working/bengaliai/results/model2020-02-23T09_33_56.405782.pt"
path_data = str(path_data.absolute())

os.environ["path_cfg"] = path_cfg
os.environ["path_weight"] = path_weight
os.environ["path_test_data"] = path_data
os.environ["PATH_DATA_RAW"] = path_data

from src.submissions.bengaliai import kaggle_project_submission

# Create the subission instance using the above yaml and weight
submission = kaggle_project_submission(input_path_CFG=path_cfg, input_path_model_weight=path_weight)

# Update the path to the pretrained model weight
submission.config.MODEL.BACKBONE.PRETRAINED_PATH = "/kaggle/working/bengaliai/models/mobilenet_v2-b0353104.pth"

# Load config and weight
submission.load()

# print(path_data)

# print(submission.model)
print(submission.config)
# print(submission.weight)

os.chdir(path_output)
submission.submit(path_data)