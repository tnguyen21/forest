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
        min_entry_len: int = -1,
        max_entry_len: int = 999,
        phonetic_representation: Callable = _dummy_phonetics,
        max_edit_distance: int = 2,
        min_jaro_winkler_sim: float = 0.0,
    ):
        """
        Add Trie to Forest with additional parameters.
        Args:
        """
        trie = Trie(max_edit_distance, min_jaro_winkler_sim)
        trie_data = {
            "trie": trie,
            "min_entry_len": min_entry_len,
            "max_entry_len": max_entry_len,
            "phonetic_representation": phonetic_representation,
        }
        self.tries.append(trie_data)

    def add_entry(self, entry: str):
        """
        Add entry to all tries in the forest
        """
        for t in self.tries:
            if len(entry) >= t["min_entry_len"] and len(entry) <= t["max_entry_len"]:
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

        * Note that words queried in the trie should be of length
        * [min_word_len - ED, max_word_len + ED]
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

            # calculate if query is valid for trie based on entry len
            query_lower_bound = t["min_entry_len"] - t["trie"].max_edit_distance
            query_upper_bound = t["max_entry_len"] + t["trie"].max_edit_distance
            if (
                len(phoneticized_word) >= query_lower_bound
                and len(phoneticized_word) <= query_upper_bound
            ):
                results = t["trie"].search(phoneticized_word)
                tentative_results.append(results)

        # ? Do we want to just return all results?
        # ? Or do we want some notion of "best" result?
        # ? Or do we want to return the best result for each trie?
        # TODO want to know where entry is coming from
        # ? figure out a way to weight similiarity scores
        # TODO do post processing to get unique results with different scores
        # ? calc both sim scores between phonetic repr and original words?
        # * ultimately asking what are entries that are similar to my query
        return tentative_results
