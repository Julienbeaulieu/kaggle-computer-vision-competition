import os
import numpy as np
import pandas as pd
import pickle
from sklearn.utils.class_weight import compute_class_weight
from dotenv import find_dotenv, load_dotenv
import click
import logging
from pathlib import Path
from typing import List
# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())

PATH_DATA = os.getenv("PATH_DATA_INTERIM")



def get_labels(input_p_data):
    """
    Retrieve the Bengali Grapheme labels of a p file based dataframe and return as a dataframe.
    :return: data frame containing the classes of the P file
    """
    classes_train = list(list(zip(*input_p_data))[1])
    df = pd.DataFrame(classes_train)
    df.columns = ["grapheme_root", "vowel_diacritic", "consonant_diacritic"]
    return df


def filter_label_df_index(df_input, index_root, index_vowel, index_consonant) -> List[int]:
    """
    Based on the inputted combo of the root, vowel, consonant, return a list of input_index within the df_input that has matching characteristics
    :param df_input:
    :param index_root:
    :param index_vowel:
    :param index_consonant:
    :return:
    """
    index_relevant = df_input.index[df_input.grapheme_root.eq(index_root) &
                                  df_input.vowel_diacritic.eq(index_vowel) &
                                  df_input.consonant_diacritic.eq(index_consonant)]
    return index_relevant


@click.command()
def compute_labeled_weights():
    # Load all data.
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


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    compute_labeled_weights()
