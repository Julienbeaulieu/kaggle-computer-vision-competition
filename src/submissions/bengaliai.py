from src.models.model_build import BuildingMixin
from src.models.model_train import TrainingMixin
from src.models.model_restore import RestorationMixin
from src.models.model_predict import PredictionMixin
import os
from src.config.config import combine_cfgs
from dotenv import find_dotenv, load_dotenv
path_CFG = os.getenv("path_cfg")
path_model_weight = os.getenv("path_weight")

class kaggle_project_submission(RestorationMixin, PredictionMixin):

    def __init__(self, input_path_CFG=path_CFG, input_path_model_weight=path_model_weight):
        """
        For kaggle project submission, must ensure to reference the YAML node which was trained on.
        :param input_CfgNode:
        """

        # Variables that will be populated by the subsequent steps.
        self.model = None
        self.results_dir = None

        # Instantiate settings
        # This overwrites the file location with .env loaded credential as well as the CFG specified here.
        self.config = combine_cfgs(path_cfg_override=input_path_CFG)

        # Null return when env not setup.
        if path_model_weight is None:
            return
        # Instantiate model, using the model weight specified, while using default self.config.
        self.restore(config=self.config, path_weight=input_path_model_weight)

        # Train the model.
        # self.train()
        self.test_eval()


        # Restore the model


        # Predict using the model
        pass


if __name__=="__main__":
    submission = kaggle_project_submission()