""" common.py
This file contains the common functions used by the single word evaluation task.
"""
""" common.py
This file contains the common functions used by the single word evaluation task.
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # noqa: E501

from trie import Trie
import pickle
import pandas as pd


def create_trie(data_path):
    """
    Loads FDA data from the given path into a Trie

    Assumes FDA data file is a csv with each word on a newline
    """
    trie = Trie()

    with open(data_path, "r") as f:
        for word in f:
            word = word.strip()
            trie.add_entry(word)
    return trie


def pickle_trie(trie, pickle_path):
    """
    Pickles the given trie to the given path
    """
    with open(pickle_path, "wb") as f:
        pickle.dump(trie, f)


def load_trie_from_pkl(pickle_path):
    """
    Loads a pickled trie from the given path
    """
    with open(pickle_path, "rb") as f:
        trie = pickle.load(f)
    return trie


if __name__ == "__main__":
    all_words_trie = create_trie("./datasets/single_word_task/dictionary.csv")

    # * The below should only need to be done once with the dataset,
    # * and then the pickled trie can be used for most pre-tasks
    # pickle_trie(all_words_trie, "./datasets/single_word_task/all_words_trie.pkl")

    test_trie = load_trie_from_pkl("./datasets/single_word_task/all_words_trie.pkl")
    print(f"Words in trie same?: {all_words_trie.entry_list == test_trie.entry_list}")
