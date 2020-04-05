from typing import List
import numpy as np
from matplotlib import pyplot as plt
import random


def vis_square(data, base_size=10):
    """
    Gallery visualizer.
    Take an array of shape (n, height, width) or (n, height, width, 3) and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    :param data:
    :return:
    """

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # Check if the data is gray or color.
    grayscale: bool = (len(data.shape) == 3)

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))

    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    fig_size = int(base_size / 8 * n)

    plt.figure(figsize=(fig_size, fig_size))

    if grayscale:
        plt.imshow(data, cmap='gray')
    else:
        plt.imshow(data)

    plt.show()


def visualize_class_imgs(data_list: List, grapheme_cls: int = None, vowel_cls: int = None, consonant_cls: int = None,
                         display_num=25):
    """

    :param data_list: list of (img, label) pairs
    :param grapheme_cls:
    :param vowel_cls:
    :param consonant_cls:
    :param display_num: number of random images to be displayed
    :return:
    """

    if grapheme_cls is not None:
        data_list = [x for x in data_list if x[1][0] == grapheme_cls]

    if vowel_cls is not None:
        data_list = [x for x in data_list if x[1][1] == vowel_cls]

    if vowel_cls is not None:
        data_list = [x for x in data_list if x[1][2] == consonant_cls]

    random.shuffle(data_list)

    imgs = np.array([x[0] for x in data_list[:display_num]])
    vis_square(imgs)
