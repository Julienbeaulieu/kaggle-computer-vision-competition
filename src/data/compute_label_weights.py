



import os
import numpy as np
import pickle
from sklearn.utils.class_weight import compute_class_weight
from dotenv import find_dotenv, load_dotenv

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())

ROOT_PATH = os.getenv("ROOT_PATH")

# Load all data.
all_data = pickle.load(open(os.path.join(ROOT_PATH, 'all_data.p'),'rb'))

# for each label load all
labels = np.array([x[1] for x in all_data])

# Get the three separate class label.
grapheme_labels =labels[:, 0]
vowel_labels =labels[:, 1]
consonant_labels =labels[:, 2]

# Use Numpy Clip to compute class weight.
consonant_weights = np.clip(compute_class_weight('balanced', list(range(np.max(consonant_labels)+1)), consonant_labels), 0.5, 3)
consonant_weights



vowel_weights = np.clip(compute_class_weight('balanced', list(range(np.max(vowel_labels)+1)), vowel_labels), 0.5, 3)
vowel_weights



grapheme_weights = np.clip(compute_class_weight('balanced', list(range(np.max(grapheme_labels)+1)), grapheme_labels), 0.5, 3)
grapheme_weights



weights  = {
    'grapheme': grapheme_weights,
    'vowel': vowel_weights,
    'consonant': consonant_weights
}



pickle.dump(weights, open(os.path.join(ROOT_PATH, 'labels_weights.p'),'wb'))



os.path.join(ROOT_PATH, 'labels_weights.p')




