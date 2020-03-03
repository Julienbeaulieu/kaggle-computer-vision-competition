from pathlib import Path

import torch
from yacs.config import CfgNode as ConfigurationNode

from src.modeling.meta_arch import build_baseline_model
from src.models.model_predict import device


class RestoreMixin:
    """
    This is a MixIn class, provid class based method to regular models.
    To REUSE this function, make sure the model class inherit from this
    MUST NOT HAVE class level data and variables.
    """

    def restore_model(self, path_cfg: ConfigurationNode, path_weight: Path):
        """
        Instantiate the model using Cfg and Weight specified.
        :param path_cfg:
        :param path_weight:
        :return:
        """
        # Use default config:
        model_config = ConfigurationNode()
        model_config.merge_from_file(path_cfg)

        if self.model is None:
            # Build model.
            self.model: torch.nn.Module = build_baseline_model(model_config)

        # Load weight dictionary
        state_dict = torch.load(path_weight, map_location='cpu')

        # Reload the model with the training dictionary.
        self.model.load_state_dict(state_dict['model_state'])

        # Send the model to device.
        self.model.to(device)