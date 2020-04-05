import pickle
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

# Load the .ENV path.
from src.data.grapheme_composition import PATH_DATA_RAW

load_dotenv(find_dotenv(), verbose=True)

# Get Env variable on the pathing.
import os
PATH_DATA_INTERIM=os.getenv("PATH_DATA_INTERIM")
PATH_DATA_RAW=os.getenv("PATH_DATA_RAW")
import pandas as pd
from typing import List


def get_image_data(p_data:List[tuple]):
    """
    Get image data component from the p data list of tuple structure
    :return:
    """
    # Convert list of tuple to tuple of list.
    tuple_of_list = list(zip(*p_data))
    tuple_of_images = tuple_of_list[0]
    list_of_images = list(tuple_of_images)
    return list_of_images

def load_label_csv():
    """
    Load the labeling data for decoding purpose
    :return:
    """

    grapheme_train = pd.read_csv(Path(PATH_DATA_RAW) / "train.csv")
    return grapheme_train

def load_data_train():
    # Load the data, ~5GB
    with open(Path(PATH_DATA_INTERIM) / "train_data.p", 'rb') as pickle_file:
        data_train = pickle.load(pickle_file)
        return data_train

def load_data_val():
    """
    Load the validation data, about 1.3GB
    """
    with open(Path(PATH_DATA_INTERIM) / "val_data.p", 'rb') as pickle_file:
        data_val = pickle.load(pickle_file)
        return data_val


def load_grapheme_classes():
    """
    Load and return the grapheme classes dataframe object
    :return:
    """
    df_grapheme_classes = pd.read_csv(Path(PATH_DATA_RAW) / "class_map.csv")
    return df_grapheme_classes