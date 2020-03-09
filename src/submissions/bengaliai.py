from ..models.model_build import BuildingMixin
from ..models.model_train import TrainingMixin
from ..models.model_restore import RestorationMixin
from ..models.model_predict import PredictionMixin
from yacs.config import CfgNode
from src.config.config import combine_cfgs

class kaggle_project_submission(BuildingMixin, TrainingMixin, RestorationMixin, PredictionMixin):

    def __init__(self, input_CfgNode: CfgNode):
        """
        For kaggle project submission, must ensure to reference the YAML node which was trained on.
        :param input_CfgNode:
        """

        # Variables that will be populated by the subsequent steps.
        self.model = None
        self.results_dir = None

        # Instantiate settings.
        self.config = input_CfgNode

        # This overwrites the file location with .env
        self.config = combine_cfgs()

        # Instantiate model.

        self.model = self.restore()

        # Train the model.
        self.train()


        # Restore the model


        # Predict using the model
        pass

