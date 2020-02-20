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

    def __init__(self):
        pass

    def __call__(self, batch: List) -> (torch.Tensor, torch.Tensor):
        """
        :param batch:
        :return:
        """
        labels = [x[1] for x in batch]
        labels = torch.tensor(labels)

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
    collator = BengaliDataBatchCollator()
    batch_size = data_cfg.BATCH_SIZE
    num_workers = min(batch_size, data_cfg.CPU_NUM)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training, collate_fn=collator,
                             num_workers=num_workers)
    return data_loader

# def rand_bbox(size, lam):
#     W = size[2]
#     H = size[3]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = np.int(W * cut_rat)
#     cut_h = np.int(H * cut_rat)

#     # uniform
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)

#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)

#     return bbx1, bby1, bbx2, bby2

# def cutmix(data, targets1, targets2, targets3, alpha):
#     indices = torch.randperm(data.size(0))
#     shuffled_data = data[indices]
#     shuffled_targets1 = targets1[indices]
#     shuffled_targets2 = targets2[indices]
#     shuffled_targets3 = targets3[indices]

#     lam = np.random.beta(alpha, alpha)
#     bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
#     data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
#     # adjust lambda to exactly match pixel ratio
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

#     targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
#     return data, targets