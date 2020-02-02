import os
import pickle
from dotenv import find_dotenv, load_dotenv

from src.config.config import cfg
from src.data.bengali_data import Preprocessor
from src.visualization.visualization import vis_square

load_dotenv(find_dotenv())
import numpy as np

PATH_DATA_INTERIM = os.getenv("PATH_DATA_INTERIM")

def visual_check_bengali():
    """
    Function called by a notebook to visually check the Bengali alphabet?
    :return:
    """
    # Load training data?
    train_data = pickle.load(open(os.path.join(PATH_DATA_INTERIM, 'val_data.p'), 'rb'))

    # Instantiate preprocessor
    preprocessor = Preprocessor(cfg.DATASET)

    # Populate image list
    imgs_list = []

    # Load 16 images
    for i in range(16):
        imgs_list.append(preprocessor(train_data[-20][0], is_training=True, normalize=False))

    # Visualize the images in a matrix format.
    vis_square(np.array(imgs_list))

