from ..models.model_build import BuildMixin
from ..models.model_train import TrainingMixin
from ..models.model_restore import RestorationMixin
from yacs.config import CfgNode
from src.config.config import combine_cfgs

class BengaliProject(BuildMixin, TrainingMixin, RestorationMixin):

    def __init__(self, input_CfgNode):
        # Variables that will be populated by the subseqeuent steps.
        self.model = None
        self.results_dir = None


        self.config = input_CfgNode
        self.config = combine_cfgs()

        # Load default path,

        # Instantiate model.

        self.model = self.build()

        # Train the model.
        self.train()


        # Restore the model


        # Predict using the model
        pass

