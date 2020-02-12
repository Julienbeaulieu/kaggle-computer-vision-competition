import numpy as np
from numpy import ndarray
from yacs.config import CfgNode
from albumentations import OneOf, Compose, MotionBlur, MedianBlur, Blur, RandomBrightnessContrast, GaussNoise, \
    GridDistortion, Rotate
from typing import Union

from cv2 import resize


def content_crop(img: ndarray) -> ndarray:
    """
    cut out the section of image where there is the most of character
    :param img: raw black white image, scale [0 to 255]
    :return: cut out img
    """
    y_list, x_list = np.where(img < 235)
    x_min, x_max = np.min(x_list), np.max(x_list)
    y_min, y_max = np.min(y_list), np.max(y_list)
    img = img[y_min:y_max, x_min:x_max]
    return img


class Preprocessor(object):
    """
    bengali data preprocessor
    """

    def __init__(self, dataset_cfg: CfgNode):
        """

        :param dataset_cfg: dataset config
        """
        aug_cfg = dataset_cfg.AUGMENTATION
        self.color_aug = self.generate_color_augmentation(aug_cfg)
        self.shape_aug = self.generate_shape_augmentation(aug_cfg)
        self.resize_shape = dataset_cfg.RESIZE_SHAPE
        self.crop = dataset_cfg.CONCENTRATE_CROP
        self.to_rgb = dataset_cfg.TO_RGB
        self.normalize_mean = dataset_cfg.get('NORMALIZE_MEAN')
        self.normalize_std = dataset_cfg.get('NORMALIZE_STD')

    @staticmethod
    def generate_color_augmentation(aug_cfg: CfgNode) -> Union[Compose, None]:
        """
        generate color augmentation object
        :param aug_cfg: augmentation config
        :return color_aug: color augmentation object
        """
        color_aug_list = []
        if aug_cfg.BRIGHTNESS_CONTRAST_PROB > 0:
            color_aug_list.append(RandomBrightnessContrast(p=aug_cfg.BRIGHTNESS_CONTRAST_PROB))

        if aug_cfg.BLURRING_PROB > 0:
            blurring = OneOf([
                MotionBlur(blur_limit=7, p=1),
                MedianBlur(blur_limit=7, p=1),
                Blur(blur_limit=7, p=1),
            ], p=aug_cfg.BLURRING_PROB)
            color_aug_list.append(blurring)

        if aug_cfg.GAUSS_NOISE_PROB > 0:
            color_aug_list.append(GaussNoise(p=aug_cfg.GAUSS_NOISE_PROB))

        if len(color_aug_list) > 0:
            color_aug = Compose(color_aug_list, p=1)
            return color_aug
        else:
            return None

    @staticmethod
    def generate_shape_augmentation(aug_cfg: CfgNode) -> Union[Compose, None]:
        """
        generate shape augmentations
        :param aug_cfg: augmentation config
        :return shape_aug: shape augmentation object
        """
        shape_aug_list = []
        if aug_cfg.ROTATION_PROB > 0:
            shape_aug_list.append(
                Rotate(limit=aug_cfg.ROTATION_DEGREE, border_mode=1, p=aug_cfg.ROTATION_PROB)
            )
        if aug_cfg.GRID_DISTORTION_PROB > 0:
            shape_aug_list.append(GridDistortion(p=aug_cfg.GRID_DISTORTION_PROB))

        if len(shape_aug_list) > 0:
            shape_aug = Compose(shape_aug_list, p=1)
            return shape_aug
        else:
            return None

    def __call__(self, img: ndarray, is_training: bool, normalize: bool = True) -> ndarray:
        """
        make the transformation
        :param img: input img array
        :param is_training: whether it's training (to do augmentation)
        :return : transformed data
        """
        x = img
        # shape augment
        if is_training and self.shape_aug is not None:
            x = self.shape_aug(image=x)['image']

        # crop
        if self.crop:
            x = content_crop(x)

        # color augment
        if is_training and self.color_aug is not None:
            x = self.color_aug(image=x)['image']

        # resize
        x = resize(x, self.resize_shape)

        # to RGB
        if self.to_rgb:
            x = np.repeat(np.expand_dims(x, axis=-1), 3, axis=-1)

        if not normalize:
            return x

        # normalize to 0-1
        x = x / 255.

        if self.normalize_mean is not None:
            x = (x - self.normalize_mean) / self.normalize_std

        return x
