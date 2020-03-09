"""
    https://arxiv.org/pdf/1901.05555.pdf
    https://github.com/richardaecn/class-balanced-loss
"""
from collections import Counter
import numpy as np
import pickle


def get_num_per_class(labels):
    """
    helper function to get number of entries per class (sorted by class input_index)
    :param labels: list of labels
    :return: array of counts sorted based on the class input_index
    """

    counts = Counter(labels)
    counts = [(x, counts[x]) for x in counts]
    counts = sorted(counts, key=lambda x: x[0])
    counts = np.array([x[1] for x in counts])
    return counts


def compute_class_balanced_weights(labels, beta: float):
    """

    :param labels: list of labels
    :param beta: make believe hyperparameter
    :return:
    """
    num_per_class = get_num_per_class(labels)
    effective_num = 1.0 - np.power(beta, num_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights)
    return weights


def compute_bengali_class_balanced_weights(full_data_path: str, beta: float, output_path: str):
    """
    compute class balanced weights on bengali dataset
    :param full_data_path: path of the pickle full dataset
    :param beta: hyperparameter for the weight algorithm
    :param output_path
    :return:
    """

    all_data = pickle.load(open(full_data_path, 'rb'))

    # for each label load all
    labels = np.array([x[1] for x in all_data])

    # Get the three separate class label.
    grapheme_labels = labels[:, 0]
    vowel_labels = labels[:, 1]
    consonant_labels = labels[:, 2]

    grapheme_weights = compute_class_balanced_weights(grapheme_labels, beta)
    vowel_weights = compute_class_balanced_weights(vowel_labels, beta)
    consonant_weights = compute_class_balanced_weights(consonant_labels, beta)

    weights = {
        'grapheme': grapheme_weights,
        'vowel': vowel_weights,
        'consonant': consonant_weights
    }
    pickle.dump(weights, open(output_path, 'wb'))
