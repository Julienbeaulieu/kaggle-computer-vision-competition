git adimport os
import numpy as np
import pickle
from sklearn.utils.class_weight import compute_class_weight
from dotenv import find_dotenv, load_dotenv
import click
import logging
from pathlib import Path

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())

PATH_DATA = os.getenv(r"PATH_DATA_INTERIM")

#@click.command()
def compute_labeled_weights():
    # Load all data.s
    all_data = pickle.load(open(os.path.join(PATH_DATA, 'all_data.p'), 'rb'))

    # for each label load all
    labels = np.array([x[1] for x in all_data])

    # Get the three separate class label.
    grapheme_labels = labels[:, 0]
    vowel_labels = labels[:, 1]
    consonant_labels = labels[:, 2]

    # Use Numpy Clip to compute the balanced class weight to ensure class balanced performance.
    consonant_weights = np.clip(
        compute_class_weight('balanced', list(range(np.max(consonant_labels) + 1)), consonant_labels), 0.5, 3)
    print(consonant_weights)

    vowel_weights = np.clip(compute_class_weight('balanced', list(range(np.max(vowel_labels) + 1)), vowel_labels), 0.5,
                            3)
    print(vowel_weights)

    grapheme_weights = np.clip(
        compute_class_weight('balanced', list(range(np.max(grapheme_labels) + 1)), grapheme_labels), 0.5, 3)
    print(grapheme_weights)

    weights = {
        'grapheme': grapheme_weights,
        'vowel': vowel_weights,
        'consonant': consonant_weights
    }

    pickle.dump(weights, open(os.path.join(PATH_DATA, 'labels_weights.p'), 'wb'))

    os.path.join(PATH_DATA, 'labels_weights.p')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt) # info = on log tout

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    compute_labeled_weights()
