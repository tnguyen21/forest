""" multiword_evaluation.py
TODO
"""

# TODO hacky way to add to python path -- need to find better way to mangage imports
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # noqa: E501

from trie import Forest
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import dill as pickle

if __name__ == "__main__":
    forest = Forest()
    dict = pd.read_csv("datasets/umls_small_dictionary/dictionary.csv")
    for _, id, term in tqdm(dict.itertuples(), desc="Processing terms..."):
        forest.add_phrase(id, term)

    forest.create_tries()
    forest.calculate_determining_scores()

    # load logistic regression model
    sample_sentence_output_path = "experiments/multi_word_evaluation_task/"
    with open(sample_sentence_output_path + "lr_model.pkl", "rb") as f:
        lr_models = pickle.load(f)

    forest.set_logistic_regression_model(lr_models)
    
    results = forest.search("Language Natural Phenomenon or Process Organic Chemical", use_lr_model=True)
    print("search text: Language Natural Phenomenon or Process Organic Chemical")
    pprint(results)