import torch
import pickle
import numpy as np
from yacs.config import CfgNode
from torch.utils.data import Dataset, DataLoader
from src.data.preprocessing import Preprocessor
from typing import List
import pandas as pd
from pathlib import Path

class BengaliDataset(Dataset):
    """
    Torch data set object for the bengali data
    """

    def __init__(self, list_data: List, node_cfg_dataset: CfgNode, is_training: bool):
        """
        :param list_data: list of raw data consists of (image, labels)
        :param node_cfg_dataset: the DATASET node of the config
        :param is_training:
        """
        self.list_image_data = list_data
        self.size_data = len(list_data)

        self.is_training = is_training

        # instantiate the preprocessor for the dataset, per the configuration node past to it.
        self.preprocessor = Preprocessor(node_cfg_dataset)

    def __len__(self) -> int:
        """
        Return the length of the dataset
        :return:
        """
        return self.size_data

    def __getitem__(self, index: int) -> (np.ndarray, np.ndarray):
        """
        Get individual item from the dataset at a particular input_index.
        :param index:
        :return:
        """
        # Retrieve both the image data as well as the labesl from the data list.
        img, labels = self.list_image_data[index]
        x = self.preprocessor(img, self.is_training)
        return x, labels


class BengaliPredictionDataset(BengaliDataset):
    """
    Torch data set object for the bengali data PREDICTION purposes only, hence much smaller and is by default assumed not to be in training
    This is a bit different from the regular dataset as the img/labels are separated.
    """

    def __init__(self, list_image_data: List, node_cfg_data: CfgNode, fname: str, indices=None):
        """
        :param list_image_data: list of raw data consists of (image, labels)
        :param node_cfg_data: data config node
        :param is_training: always false in the prediction dataset.
        """
        super().__init__(list_image_data, node_cfg_data, is_training=False)
        self.list_image_data = list_image_data
        self.size_data = len(list_image_data)

        # Record the indices if that is not provided.
        if indices is None:
            indices = np.arange(self.size_data)
        self.indices = indices

        # Read the data frame. This will be used during get-item phase.
        self.df = pd.read_parquet(fname)

        self.is_training = False

        # instantiate the preprocessor for the dataset, per the configuration node past to it.
        self.preprocessor = Preprocessor(node_cfg_data)

    def __len__(self) -> int:
        """
        Instead of Returning the length of the dataset, now it returns the length of the indices
        :return:
        """
        return len(self.indices)

    def __getitem__(self, input_index: int) -> (np.ndarray, np.ndarray):
        """
        Get individual item from the dataset at a particular index.
        :param input_index:
        :return:
        """
        # Retrieve the input_index among indices:
        index = self.indices[input_index]

        # Retrieve both the image data as well as the labesl from the data list.
        img = self.list_image_data[index]

        # Preprocess the image through the pre
        augmented_image = self.preprocessor(img, self.is_training)

        # Get the name from the spreadsheet:
        name = self.df.iloc[input_index, 0]
        return augmented_image, name

class BengaliDataBatchCollator(object):
    """
    Custom collator
    """

    def __init__(self, is_training, do_augmix):
        self.is_training = is_training
        self.do_augmix = do_augmix

    def __call__(self, batch: List) -> (torch.Tensor, torch.Tensor):
        """
        :param batch:
        :return:
        """
        labels = [x[1] for x in batch]
        labels = torch.tensor(labels)

        if self.do_augmix and self.is_training:
            inputs = batch[0]
            inputs = np.array([x[0] for x in inputs])
            inputs_aug1 = np.array([x[1] for x in inputs])
            inputs_aug2 = np.array([x[2] for x in inputs])
            inputs = np.vstack([inputs, inputs_aug1, inputs_aug2])
        else:
            inputs = np.array([x[0] for x in batch])

        inputs = torch.tensor(inputs)
        inputs = inputs.permute([0, 3, 1, 2])
        return inputs, labels


def build_data_loader(data_list: List, data_cfg: CfgNode, is_training: bool) -> DataLoader:
    """
    generate data loader
    :param data_list: list of (img, labels)
    :param data_cfg: data config node/
    :param is_training: whether training
    :return: data loader
    """
    dataset = BengaliDataset(data_list, data_cfg, is_training)
    collator = BengaliDataBatchCollator(is_training, data_cfg.DO_AUGMIX)
    batch_size = data_cfg.BATCH_SIZE

    # limit the number of works based on CPU number.
    num_workers = min(batch_size, data_cfg.CPU_NUM)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training, collate_fn=collator,
                             num_workers=num_workers)
    return data_loader

