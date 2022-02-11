""" common.py
This file contains the common functions used by the single word evaluation task.
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # noqa: E501

from trie import Trie, PhoneticTrie
from phonetics import metaphone, dmetaphone
from jellyfish import soundex
from fuzzy import nysiis
import dill as pickle
import pandas as pd


def create_trie(data_path):
    """
    Loads FDA data from the given path into a Trie

    Assumes FDA data file is a csv with each word on a newline
    """
    #! create multiple tries with different phonetic representations
    #! and then pickle them
    trie = Trie()

    with open(data_path, "r") as f:
        for word in f:
            word = word.strip()
            trie.add_entry(word)
    return trie


def create_phonetic_trie(data_path):
    """
    Loads FDA data from the given path into a Trie

    Assumes FDA data file is a csv with each word on a newline
    """
    ptrie = PhoneticTrie()

    # Add trie with default values
    ptrie.add_trie()

    with open(data_path, "r") as f:
        for word in f:
            word = word.strip()
            ptrie.add_entry(word)
    return ptrie


def create_phonetic_trie_all_phonetics(data_path):
    """
    Loads FDA data from the given path into multiple tries
    Each trie uses a different phonetic algorithm
    """
    ptrie = PhoneticTrie()

    # Add trie with default values
    ptrie.add_trie()

    # Add trie with dmetaphone
    # dmetaphone returns two phonetic representations, take the first
    dmetaphone_wrapper = lambda word: dmetaphone(word)[0]
    ptrie.add_trie(phonetic_representation=dmetaphone_wrapper)

    # Add trie with metaphone
    ptrie.add_trie(phonetic_representation=metaphone)

    # Add trie with nysiis
    ptrie.add_trie(phonetic_representation=nysiis)

    # Add trie with soundex
    ptrie.add_trie(phonetic_representation=soundex)

    with open(data_path, "r") as f:
        for word in f:
            word = word.strip()
            ptrie.add_entry(word)
    return ptrie


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
    all_words_ptrie = create_phonetic_trie_all_phonetics(
        "./datasets/single_word_task/dictionary.csv"
    )

    # * The below should only need to be done once with the dataset,
    # * and then the pickled trie can be used for most pre-tasks
    pickle_trie(
        all_words_ptrie, "./datasets/single_word_task/all_words_all_phonetics_ptrie.pkl"
    )

    test_ptrie = load_trie_from_pkl(
        "./datasets/single_word_task/all_words_all_phonetics_ptrie.pkl"
    )
    print(
        f"Words in trie same?: {all_words_ptrie.tries[0]['trie'].entry_list == test_ptrie.tries[0]['trie'].entry_list}"
    )
