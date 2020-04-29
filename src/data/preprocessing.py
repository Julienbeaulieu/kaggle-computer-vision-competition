import numpy as np
from numpy import ndarray
from yacs.config import CfgNode
from albumentations import OneOf, Compose, MotionBlur, MedianBlur, Blur, RandomBrightnessContrast, GaussNoise, \
    GridDistortion, Rotate, HorizontalFlip, CoarseDropout, Cutout
from .grid_mask import GridMask
from typing import Union, List, Tuple
from .augmix import augmentations, augment_and_mix
import cv2
from cv2 import resize
import torch

def content_crop(img: ndarray, white_background: bool):
    """
    Center the content, removed
    https://www.kaggle.com/iafoss/image-preprocessing-128x128

    :param img: grapheme image matrix
    :param white_background: whether the image
    :return: cropped image matrix
    """
    # Remove the surrounding 5 pixels
    img = img[5:-5, 5:-5]
    if white_background:
        y_list, x_list = np.where(img < 235)
    else:
        y_list, x_list = np.where(img > 80)

    # get xy min max
    xmin, xmax = np.min(x_list), np.max(x_list)
    ymin, ymax = np.min(y_list), np.max(y_list)

    # Manually set the baseline low and high for x&y
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < 223) else 236
    ymax = ymax + 10 if (ymax < 127) else 137

    # Reposition the images
    img = img[ymin:ymax, xmin:xmax]

    return img


def pad_to_square(img: ndarray, white_background: bool):
    ly, lx = img.shape

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

    def __init__(self, node_cfg_dataset: CfgNode):
        """
        Constructor of the Preprocessing from the Configuration Node properties.
        :param node_cfg_dataset: dataset config
        """
        # Augmentation node is the
        aug_cfg = node_cfg_dataset.AUGMENTATION


        # !!!Training ONLY!!!
        # Color augmentation settings,
        self.color_aug = self.generate_color_augmentation(aug_cfg)
        # Shape augmentation settings
        self.shape_aug = self.generate_shape_augmentation(aug_cfg)
        # Cutout augmentation settings
        self.cutout_aug = self.generate_cutout_augmentation(aug_cfg)
        self.pad = node_cfg_dataset.PAD_TO_SQUARE
        self.white_background = node_cfg_dataset.WHITE_BACKGROUND
        self.do_augmix = node_cfg_dataset.DO_AUGMIX

        # !!!~~~BOTH~~~!!!
        # Color augmentation settings,
        self.resize_shape = node_cfg_dataset.RESIZE_SHAPE
        # Crop augmentation settings,
        self.crop = node_cfg_dataset.CONCENTRATE_CROP
        # Convert to RGB
        self.to_rgb = node_cfg_dataset.TO_RGB
        # Normalize Mean or STD?
        self.normalize_mean = node_cfg_dataset.get('NORMALIZE_MEAN')
        self.normalize_std = node_cfg_dataset.get('NORMALIZE_STD')


        if self.do_augmix:
            augmentations.IMAGE_SIZE = node_cfg_dataset.RESIZE_SHAPE[0]

        if not self.to_rgb:
            self.normalize_mean = np.mean(self.normalize_mean)
            self.normalize_std = np.mean(self.normalize_std)

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
                MotionBlur(aug_cfg.BLUR_LIMIT, p=1),
                MedianBlur(aug_cfg.BLUR_LIMIT, p=1),
                Blur(aug_cfg.BLUR_LIMIT, p=1),
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
    
    @staticmethod
    def generate_cutout_augmentation(aug_cfg: CfgNode):
        
        cutout_aug_list = []
        if aug_cfg.CUTOUT_PROB > 0:
            cutout_aug_list.append(Cutout(num_holes=1, max_h_size=aug_cfg.HEIGHT//2, max_w_size=aug_cfg.WIDTH//2, 
                                        fill_value=255, p=aug_cfg.CUTOUT_PROB))
                                  
        if len(cutout_aug_list) > 0:
            cutout_aug = Compose(cutout_aug_list, p=1)
            return cutout_aug
        else:
            return None  

    def __call__(self, img: ndarray, is_training: bool, normalize: bool = True) -> Union[ndarray, Tuple]:
        """
        Conduct the transformation
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

        # shape augment
        if is_training and self.shape_aug is not None:
            x = self.shape_aug(image=x)['image']

        # color & cutout augment
        if is_training and self.color_aug is not None:
            x = self.color_aug(image=x)['image']
            x = self.cutout_aug(image=x)['image']

        if not normalize:
            return x
        
        x = self.normalize_img(x)
        return x

        x = self.normalize_img(x)

        # Resume the permutation
        img = torch.tensor(x)
        img = img.permute([2, 0, 1])

        return img

    def normalize_img(self, x: ndarray) -> ndarray:
        """
        Normalize image to a specific mean/std if they are specifiied, otherwise, default to /255
        :param x:
        :return:
        """
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
