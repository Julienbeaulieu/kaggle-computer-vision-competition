import numpy as np
import os
import sys
import pickle
from PIL import Image
from cv2 import resize
import time
from matplotlib import pyplot as plt
sys.path.append('../')
import torch
from src.config.config import get_cfg_defaults
from src.modeling.backbone.build import build_backbone, BACKBONE_REGISTRY
from src.modeling.meta_arch.baseline import build_baseline_model
from yacs.config import CfgNode as ConfigurationNode

class BuildMixin:

    def build(self,
              config: ConfigurationNode = None,
              path_yaml: str = r"C:\Git\bengali.ai\configs\DevYang.yaml"
              ):
        """
        This mixin function build the models using the information provided.
        :return:
        """

        # No argument, use default class object.
        if config is None:
            config = self.config

        if self.model is None:
            # Specify how the model will be build.
            self.model = build_baseline_model(config.MODEL)

        # Generate randomized input tensor to test size.
        inputs = torch.rand(2, 3, 128, 128)

        # Get an estimation of the output size. 2
        outputs = self.model(inputs)

        print("===================")
        print("Backbone MODULES:")
        print("===================")

        # Print out all the model backbone modules.
        for x in self.model.backbone.modules():
            print(x)

        print("===================")
        print("HEAD MODULES:")
        print("===================")

        # Print out all the model backbone modules.
        for y in self.model.head.modules():
            print(y)
