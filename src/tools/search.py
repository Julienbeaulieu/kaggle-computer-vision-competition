from pathlib import Path
import os
import re

path_default_dir = Path(__file__).parents[2] / "results"
#print(path_default_dir)

def is_yaml(input_str):
	"""
	Function used to screen for config yaml files.
	:param string:
	:return:
	"""
	p = re.compile("^config.*yaml$")
	if p.search(input_str):
		return True
	return False

def is_model(input_str):
	p = re.compile("^model.*pt$")
	if p.search(input_str):
		return True
	return False

def get_first_yaml(path_input: Path=path_default_dir):
	"""
	Retrieve the first yaml file from the path,
	Used mostly to get the best results.
	:param path_input:
	:return:
	"""
	list_yamls = list(filter(is_yaml, os.listdir(path_input)))
	if len(list_yamls) > 0:
		return path_default_dir / list_yamls[0]
	else:
		return None

def get_first_model(path_input: Path=path_default_dir):
	"""
	Retrieve the first yaml file from the path,
	Used mostly to get the best results.
	:param path_input:
	:return:
	"""

	list_models = list(filter(is_model, os.listdir(path_input)))
	if len(list_models) > 0:
		return path_default_dir / list_models[0]
	else:
		return None

if __name__=="__main__":
	print(get_first_model())
	print(get_first_yaml())