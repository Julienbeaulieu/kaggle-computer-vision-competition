import sys
from torch.utils.data import Dataset, DataLoader
from src.data.bengali_data import BengaliDataset, BengaliPredictionDataset
from src.data.make_dataset import load_images
sys.path.append('../')
from torch import nn
import torch
import gc
import pandas as pd
import os

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"  # can also be cuda
import warnings
from pathlib import Path

class PredictionMixin:
    """
    This is a MixIn class, provid class based method to regular models.
    To REUSE this function, make sure the model class inherit from this
    MUST NOT HAVE class level data and variables.
    """


    def test_eval(self, DataDir: Path or str):
        """
        This is adapted from Julien's test inference script from Kaggle.
        Assuming a model_restore is run and model predict is to be tested.
        :return:
        """
        if self.debug:
            print("assert model")
        assert self.model is not None

        if self.debug:
            print("assert config")
        assert self.config is not None

        if self.debug:
            print("assert_datadir")
        if not os.path.isdir(DataDir):
            raise ValueError("A valid data directory MUST be specified")

        if self.debug:
            print("assert_model")
        self.model.eval()

        #  list of test data.
        test_data = ['test_image_data_0.parquet',
                     'test_image_data_1.parquet',
                     'test_image_data_2.parquet',
                     'test_image_data_3.parquet']

        row_id, target = [], []

        batch_size = 1

        for index, name_file in enumerate(test_data):

            # Get the image data
            test_images = load_images('test', indices=[str(index)])
            if self.debug:
                print("forming file path")
            # This is the full path to the testfile
            path_file = Path(DataDir) / name_file

            # Construct the DataSet from the images and file name.
            test_dataset = BengaliPredictionDataset(test_images, self.config.DATASET, fname=path_file.absolute())

            # Construct the data loader from the data set.
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=self.config.DATASET.CPU_NUM, shuffle=False)

            with torch.no_grad():
                # look through the test loader, which has only a SINGLE unit.
                for inputs, name in test_loader:

                    # Push to device, CPU or GPU.
                    inputs = inputs.to(device)

                    # Get name, and strip various ?
                    name = str(name).strip("'(),'")

                    #  Get the Logitc numbers from the floats returned.
                    grapheme_logits, vowel_logits, consonant_logits = self.model(inputs.float())

                    # Return max value?
                    grapheme_logits = grapheme_logits.argmax(-1)

                    # Return max value?
                    vowel_logits = vowel_logits.argmax(-1)

                    # Return max value?
                    consonant_logits = consonant_logits.argmax(-1)

                    # use a for loop if batch_size > 1
                    row_id += [f'{name}_grapheme_root',
                               f'{name}_vowel_diacritic',
                               f'{name}_consonant_diacritic']
                    target += [grapheme_logits.item(),
                               vowel_logits.item(),
                               consonant_logits.item()]

                del test_images, test_dataset, test_loader
                gc.collect()

        return pd.DataFrame({'row_id': row_id, 'target': target})

    def predict(self):
        if self.model is None:
            warnings.warn("This project object has no model in it, prediction and evaluation cannot commence")
            return

        self.model.eval()