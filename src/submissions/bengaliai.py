from src.models.model_restore import RestorationMixin
from src.models.model_predict import PredictionMixin
from src.tools.search import get_first_yaml, get_first_model
import os
from src.config.config import combine_cfgs
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from shutil import copytree, ignore_patterns
import zipfile

load_dotenv(find_dotenv())

#path_CFG = os.getenv("path_cfg")
#path_model_weight = os.getenv("path_weight")
# Get the path to the raw data folder for testing.
path_test_data = os.getenv("path_test_data")

class kaggle_project_submission(RestorationMixin, PredictionMixin):
    """
    This is the submission class to be instantiated with the right information on Kaggle notebook in order to produce a submission.csv.
    """

    def __init__(self, input_path_CFG=get_first_yaml(), input_path_model_weight=get_first_model(), debug=True):
        """
        For kaggle project submission, must ensure to reference the YAML node which was trained on.
        :param input_CfgNode:
        """

        # Variables that will be populated by the subsequent steps.
        self.model = None
        self.results_dir = None
        self.debug = debug

        # Instantiate settings
        # This overwrites the file location with .env loaded credential as well as the CFG specified here.
        self.config = combine_cfgs(path_cfg_override=input_path_CFG)
        self.weight = input_path_model_weight

    def generate_kaggle_upload(self, path_out: Path = Path(r"C:\temp\test\\")):
        """
        This ensure only the absolutely necessary data are copied over to a temporary location, ignoring everything.

        This generate a filder list and past on to be ignored by the copytree operations

        :return:
        """
        pattern_exclusion_files = ignore_patterns("*.p",
                                                  "*.md",
                                                  "*.pkl",
                                                  "*.zip",
                                                  "wheelhouse",
                                                  "train*.parquet",
                                                  "*.ipynb",
                                                  ".pytest*",
                                                  ".git",
                                                  "model_bak*.pt",
                                                  ".idea",
                                                  "reports",
                                                  "references",
                                                  "docs",
                                                  "2020-02*",
                                                  "src.egg-info",
                                                  "mlruns",
                                                  "test_*.py",
                                                  "*.md",
                                                  "submission.csv",
                                                  "notebooks",)

        # Go tot
        path_root = Path(__file__).parents[2]

        copytree(path_root, path_out, ignore=pattern_exclusion_files)

    def generate_kaggle_upload_zip(self, path_out: Path = Path(r"C:\temp\test\\")):
        self.generate_kaggle_upload(path_out)

        # Source; https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory-in-python

        # Generate zip file handle,
        zipf = zipfile.ZipFile(path_out.parent / 'Submission.zip', 'w', zipfile.ZIP_DEFLATED)

        # Call the zipping function using the zip handle.
        zipdir(path_out.absolute(), zipf)

        # Close zip file.
        zipf.close()

    def load(self, input_config=None, input_weight=None):
        """
        Load the config and weights necessary for the model, based on class instantiation yet allow overwritting and modifying.
        :param input_config:
        :param input_weight:
        :return:
        """
        # Default paramemters to use self config/weight parameters.
        # yet flexible enough to allow overwriting.
        if input_config is None:
            input_config = self.config

        if input_weight is None:
            input_weight = self.weight

        # Instantiate model, using the model weight specified, while using default self.config.
        self.restore(config=input_config, path_weight=input_weight)

    def submit(self, path_input=path_test_data):
        """
        Test predicting using the model
        :return:
        """
        try:
            self.test_eval(path_input).to_csv('submission.csv', index=False)
        except Exception as e:
            print(e)


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(
                           os.path.join(root, file),
                           os.path.join(path, '.'))
                       )



if __name__=="__main__":
    # Instantiate class
    submission = kaggle_project_submission()

    # Load default (using .env)
    submission.load()
    submission.submit()
    submission.generate_kaggle_upload_zip()