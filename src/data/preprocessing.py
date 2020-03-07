import numpy as np
from numpy import ndarray
from yacs.config import CfgNode
from albumentations import OneOf, Compose, MotionBlur, MedianBlur, Blur, RandomBrightnessContrast, GaussNoise, \
    GridDistortion, Rotate, HorizontalFlip
from .grid_mask import GridMask
from typing import Union, List, Tuple
from .augmix import augmentations, augment_and_mix

from cv2 import resize


def content_crop(img: ndarray, white_background: bool):
    """

    https://www.kaggle.com/iafoss/image-preprocessing-128x128

    :param img: grapheme image matrix
    :param white_background: whether the image
    :return: cropped image matrix
    """
    # remove the surrounding 5 pixels
    img = img[5:-5, 5:-5]
    if white_background:
        y_list, x_list = np.where(img < 235)
    else:
        y_list, x_list = np.where(img > 80)

    # get xy min max
    xmin, xmax = np.min(x_list), np.max(x_list)
    ymin, ymax = np.min(y_list), np.max(y_list)

    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < 223) else 236
    ymax = ymax + 10 if (ymax < 127) else 137
    img = img[ymin:ymax, xmin:xmax]

    return img


def pad_to_square(img: ndarray, white_background: bool):
    ly, lx, _ = img.shape

    l = max(lx, ly) + 16
    if white_background:
        constant_pad = 255
    else:
        constant_pad = 0
    img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode='constant', constant_values=constant_pad)
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
        self.pad = dataset_cfg.PAD_TO_SQUARE
        self.white_background = dataset_cfg.WHITE_BACKGROUND
        self.to_rgb = dataset_cfg.TO_RGB
        self.normalize_mean = dataset_cfg.get('NORMALIZE_MEAN')
        self.normalize_std = dataset_cfg.get('NORMALIZE_STD')

        self.do_augmix = dataset_cfg.DO_AUGMIX
        if self.do_augmix:
            augmentations.IMAGE_SIZE = dataset_cfg.RESIZE_SHAPE[0]

        if not self.to_rgb:
            self.normalize_mean = np.mean(self.normalize_mean)
            self.normalize_std = np.mean(self.normalize_std)

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
        if aug_cfg.GRID_MASK_PROB > 0:
            color_aug_list.append(GridMask(num_grid=(3, 7), p=aug_cfg.GRID_MASK_PROB))
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
        if aug_cfg.HORIZONTAL_FLIP_PROB > 0:
            shape_aug_list.append(HorizontalFlip(p=aug_cfg.HORIZONTAL_FLIP_PROB ))
        if len(shape_aug_list) > 0:
            shape_aug = Compose(shape_aug_list, p=1)
            return shape_aug
        else:
            return None

    def __call__(self, img: ndarray, is_training: bool, normalize: bool = True) -> Union[ndarray, Tuple]:
        """
        make the transformation
        :param img: input img array
        :param is_training: whether it's training (to do augmentation)
        :return : transformed data
        """
        x = img

        # if not white background, reverse
        if not self.white_background:
            x = 255 - x
        # crop
        if self.crop:
            x = content_crop(x, self.white_background)
        if self.pad:
            x = pad_to_square(x, self.white_background)
        # resize
        x = resize(x, self.resize_shape)
        # to RGB
        if self.to_rgb:
            x = np.repeat(np.expand_dims(x, axis=-1), 3, axis=-1)
        else:
            x = np.expand_dims(x, axis=-1)

        # shape augment
        if is_training:
            if self.do_augmix:
                return self.compute_augmix_inputs(img)
            else:
                # normal shape color changes
                if self.shape_aug is not None:
                    x = self.shape_aug(image=x)['image']

                if self.color_aug is not None:
                    x = self.color_aug(image=x)['image']

        if not normalize:
            return x

        x = self.normalize_img(x)
        return x

    def normalize_img(self, x: ndarray) -> ndarray:
        # normalize to 0-1
        x = x / 255.
        if self.normalize_mean is not None:
            x = (x - self.normalize_mean) / self.normalize_std
        return x

    def compute_augmix_inputs(self, img):
        aug1 = augment_and_mix(img)
        aug1 = self.normalize_img(aug1)
        aug2 = augment_and_mix(img)
        aug2 = self.normalize_img(aug2)

        img = self.normalize_img(img)
        return img, aug1, aug2
