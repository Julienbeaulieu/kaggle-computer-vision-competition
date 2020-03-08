from pathlib import Path

import torch
from yacs.config import CfgNode as ConfigurationNode

from src.modeling.meta_arch import build_baseline_model
from src.models.model_predict import device


class RestorationMixin:
    """
    This is a MixIn class, provid class based method to regular models.
    To REUSE this function, make sure the model class inherit from this
    MUST NOT HAVE class level data and variables.
    """

    def restore(self, path_weight: Path, config: ConfigurationNode = None):
        """
        Instantiate the model using Cfg and Weight specified.
        :param config:
        :param path_weight:
        :return:
        """
        # Use Class default config:
        if config is None:
            config = self.config

        if self.model is None:
            # Build model.
            self.model: torch.nn.Module = build_baseline_model(config)

        # Load weight dictionary
        state_dict = torch.load(path_weight, map_location='cpu')

        # Reload the model with the training dictionary.
        self.model.load_state_dict(state_dict['model_state'])

        # Send the model to device.
        self.model.to(device)