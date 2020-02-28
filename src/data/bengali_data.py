import torch
import pickle
import numpy as np
from yacs.config import CfgNode
from torch.utils.data import Dataset, DataLoader
from .preprocessing import Preprocessor
from typing import List


class BengaliDataset(Dataset):
    """
    Torch data set object for the bengali data
    """

    def __init__(self, data_list: List, data_cfg: CfgNode, is_training: bool):
        """
        :param data_list: list of raw data consists of (image, labels)
        :param data_cfg:  data config node
        :param is_training:
        """
        self.data_list = data_list
        self.data_size = len(data_list)
        self.is_training = is_training
        self.preprocessor = Preprocessor(data_cfg)

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        img, labels = self.data_list[idx]
        x = self.preprocessor(img, self.is_training)
        return x, labels


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
    :param data_cfg: data config node
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
