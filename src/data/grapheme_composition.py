from dotenv import load_dotenv, find_dotenv
import os
# Load the .ENV path.


load_dotenv(find_dotenv())
PATH_DATA_RAW = os.getenv("PATH_DATA_RAW")
from typing import List


def encode_grapheme(index_root: int, index_vowel: int, index_consonant: int):
    """
    Takes the components and return the combined graphemes
    :return:
    """
    from src.data.load_datasets import load_label_csv
    grapheme_train = load_label_csv()
    # Look up the entries that have the same classifications.
    d = grapheme_train[grapheme_train.grapheme_root.eq(index_root) &
                       grapheme_train.vowel_diacritic.eq(index_vowel) &
                       grapheme_train.consonant_diacritic.eq(index_consonant)]

    if len(d) != 0:
        # Show the first one and its grapheme (since they all should have the same grapheme)
        from IPython.display import display, Markdown
        display(Markdown('<h1>{}</h1>'.format(f"Character{d.iloc[0].grapheme}")))
        # print(f"Character{d.iloc[0].grapheme}")

def get_components(input_list: List[int]) -> List[str]:
    """
    Based on the input list of three components, return
    :param input_list: must have three integer parts for root, vowel and cosonant diacritic input_index
    :return:
    """
    from src.data.load_datasets import load_grapheme_classes
    # The input list must have 3
    assert len(input_list) == 3

    # Load the decoder dataframe.
    grapheme_classes = load_grapheme_classes()

    # Get the respective components
    component1 = grapheme_classes[grapheme_classes.component_type.eq("grapheme_root") & grapheme_classes.label.eq(input_list[0])].iloc[0].component
    component2 = grapheme_classes[grapheme_classes.component_type.eq("vowel_diacritic") & grapheme_classes.label.eq(input_list[1])].iloc[0].component
    component3 = grapheme_classes[grapheme_classes.component_type.eq("consonant_diacritic") & grapheme_classes.label.eq(input_list[2])].iloc[0].component

    # return the components
    return [component1, component2, component3]