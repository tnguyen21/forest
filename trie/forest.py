""" trie/forest.py

Forest data structure which contains a list of Tries, each with different
hyperparameters (e.g. ED thresholds, phonetic representation, word length limits)

Provides interface to add tries and entries to the Forest, and conduct queries
across all Tries in the forest.

"""
from typing import Callable, List, Tuple
from .trie import Trie

# ? How do we store multiple dictionaries and keep phonetic representations of them?
# ? How do we store multiple phonetic representations of the same dictionary?
# ? How do we store these within Tries?


class Forest:
    def __init__(self):
        self.tries = []
        self.phonetic_map = {}

    def _dummy_phonetics(word: str) -> str:
        """
        Dummy phonetic representation to be
        """
        return word

    def add_trie(
        self,
        trie: Trie,
        min_word_len: int,
        max_word_len: int,
        phonetic_representation: Callable = _dummy_phonetics,
        max_edit_distance: int = 2,
        min_jaro_winkler_sim: float = 0.0,
    ):
        """
        Add Trie to Forest with additional parameters
        Args:
        """
        trie = Trie(max_edit_distance, min_jaro_winkler_sim)
        trie_data = {
            "trie": trie,
            "min_word_len": min_word_len,
            "max_word_len": max_word_len,
            "phonetic_representation": phonetic_representation,
        }
        self.tries.append(trie_data)

    def add_entry(self, entry: str):
        """
        Add entry to all tries in the forest
        """
        for t in self.tries:
            if len(entry) >= t["min_word_len"] and len(entry) <= t["max_word_len"]:
                # check if phonetic function is in our map
                if t["phonetic_representation"] not in self.phonetic_map:
                    self.phonetic_map[t["phonetic_representation"]] = {}

                # convert entry into phonetic representation (if any is needed)
                phoneticized_entry = t["phonetic_representation"](entry)

                # check if phoneticized entry exist in phonetic map
                if (
                    phoneticized_entry
                    not in self.phonetic_map[t["phonetic_representation"]]
                ):
                    self.phonetic_map[t["phonetic_representation"]][
                        phoneticized_entry
                    ] = []

                # add original entry to phonetic map
                self.phonetic_map[t["phonetic_representation"]][
                    phoneticized_entry
                ].append(entry)

                t["trie"].add_entry(phoneticized_entry)

    def search(self, word: str) -> List[Tuple[str, int, float]]:
        """
        Conduct search with query word in parallel on all
        tries in the trie
        Args:
            word: query word
        Return:
            list of tuples that has potential matches of query with
                edit distance and jaro-winkler similiarity
        """
        tentative_results = []

        # search each trie for the word
        for i, t in enumerate(self.tries):
            print(f"Searching Trie {i}/{len(self.tries)}")

            # convert word into phonetic representation (if any is needed)
            phoneticized_word = t["phonetic_representation"](word)

            results = t["trie"].search(phoneticized_word)
            tentative_results.append(results)

        # ? Do we want to just return all results?
        # ? Or do we want some notion of "best" result?
        # ? Or do we want to return the best result for each trie?
        return tentative_results
