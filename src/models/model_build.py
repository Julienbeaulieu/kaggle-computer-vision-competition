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



cfg = get_cfg_defaults()
cfg.merge_from_file(r"C:\Git\bengali.ai\configs\DevYang.yaml")

# Specify how the model will be build.
model = build_baseline_model(cfg.MODEL)

# Generate randomized input tensor to test size.
inputs = torch.rand(2, 3, 128, 128)

# Get an estimation of the output size. 2
outputs = model(inputs)

# Print out all the model backbone modules.
for x in model.backbone.modules():
    print(x)

print("===================")
print("HEAD MODULES:")
print("===================")

# Print out all the model backbone modules.
for y in model.head.modules():
    print(y)
