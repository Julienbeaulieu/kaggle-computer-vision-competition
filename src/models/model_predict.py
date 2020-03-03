import sys

sys.path.append('../')
from torch import nn

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
import warnings

class Evaluation:
    """
    This is a MixIn class, provid class based method to regular models.
    To REUSE this function, make sure the model class inherit from this
    MUST NOT HAVE class level data and variables.
    """

    def evaluate_model(self):
        if self.model is None:
            warnings.warn("This project object has no model in it, prediction and evaluation cannot commence")
            return

        self.model.eval()

        # Load test data.
        test_data = ['test_image_data_0.parquet', 'test_image_data_1.parquet', 'test_image_data_2.parquet',
                     'test_image_data_3.parquet']


        row_id, target = [], []

        batch_size = 1
        for index, file_name in enumerate(test_data):

            test_images = get_data('test', indices=[str(index)])

            test_dataset = BengaliDataset(test_images, dataset_cfg, fname=file_name)

            # test_collator = BengaliDataBatchCollator() ---> don't need batch collator for batch size of 1

            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=4)

            with torch.no_grad():
                for inputs, name in test_loader:
                    inputs = inputs.to(device)
                    name = str(name).strip("'(),'")
                    grapheme_logits, vowel_logits, consonant_logits = self.model(inputs.float())

                    grapheme_logits = grapheme_logits.argmax(-1)
                    vowel_logits = vowel_logits.argmax(-1)
                    consonant_logits = consonant_logits.argmax(-1)

                    # use a for loop if batch_size > 1
                    row_id += [f'{name}_grapheme_root', f'{name}_vowel_diacritic',
                               f'{name}_consonant_diacritic']
                    target += [grapheme_logits.item(), vowel_logits.item(),
                               consonant_logits.item()]
                del test_images, test_dataset, test_loader
                gc.collect()

        return pd.DataFrame({'row_id': row_id, 'target': target})