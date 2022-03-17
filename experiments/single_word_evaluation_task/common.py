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
from tqdm import tqdm
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
        # skip first line because header
        lines = f.readlines()
        lines = lines[1:]
        for word in lines:
            word = word.strip()
            ptrie.add_entry(word)
    return ptrie


def create_phonetic_trie_all_phonetics(data_path):
    """
    Loads data from the given path into multiple tries
    Each trie uses a different phonetic algorithm

    Assumes data is a dictionary with words separated by newlines
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
        # skip first line because header
        lines = f.readlines()
        lines = lines[1:]

        for word in lines:
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


def generate_data_set(
    ptrie: PhoneticTrie, data_df: pd.DataFrame, edit_distance: int = 2
) -> pd.DataFrame:
    """
    Args:
        ptrie: PhoneticTrie to use for searching
        data_df: DataFrame containing the data to be searched
            which has columns ["word", "search", "edit_distance"]
    """
    train_columns = [
        "target_word",
        "result_word",
        "query",
        "dmetaphone_sim",
        "dmetaphone_ed",
        "metaphone_sim",
        "metaphone_ed",
        "nysiis_sim",
        "nysiis_ed",
        "soundex_sim",
        "soundex_ed",
        "og_sim",
        "og_ed",
        "label",
    ]

    to_df_list = []
    for idx, word, search, ed in tqdm(
        data_df.itertuples(), ascii=True, desc="Generating data set"
    ):
        results = ptrie.search(
            search,
            max_edit_distance=edit_distance,
            metaphone_output=True,
            dmetaphone_output=True,
            soundex_output=True,
            nysiis_output=True,
            use_lr_model=False,
        )
        for result in results:
            #! if result is empty, should still add a row when creating a dataset
            #! failed searched
            label = 1 if result["result"] == word else 0

            append_row = [
                word,
                result["result"],
                search,
                result["dmetaphone_jaro_winkler_similarity"],
                result["dmetaphone_edit_distance"],
                result["metaphone_jaro_winkler_similarity"],
                result["metaphone_edit_distance"],
                result["nysiis_jaro_winkler_similarity"],
                result["nysiis_edit_distance"],
                result["soundex_jaro_winkler_similarity"],
                result["soundex_edit_distance"],
                result["original_jaro_winkler_similarity"],
                result["original_edit_distance"],
                label,
            ]

            # append row to df
            to_df_list.append(append_row)

    train_df = pd.DataFrame(to_df_list, columns=train_columns)

    return train_df


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
