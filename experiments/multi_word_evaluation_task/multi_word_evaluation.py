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
    # nasa_df = pd.read_csv("datasets/nasa_shared_task/HEXTRATO_dictionary.csv")
    # for _, id, title in tqdm(nasa_df.itertuples(), desc="Processing NASA shared task dictionary..."):
        forest.add_phrase(id, title)

    forest.create_tries()
    forest.calculate_determining_scores()

    # results = forest.get_token_concept_dictionary("Bad Man with Gun Tommy Nguyen Bad Man with Gun")
    results = forest.search("Bad Man with Gun Tommy Nguyen Bad Man with Gun")
    # print(results)
    for _ in results:
        print(_)