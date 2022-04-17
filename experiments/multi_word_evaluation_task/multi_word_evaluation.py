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

if __name__ == "__main__":
    forest = Forest()
    imbd_df = pd.read_csv("datasets/imdb_movie_titles/-a.csv")
    for _, id, title in tqdm(imbd_df.itertuples(), desc="Processing IMDB titles..."):
        forest.add_phrase(id, title)
    
    pprint(forest.concept_id_expression_gazetteer)
    pprint(forest.word_expression_gazetteer)