""" trie/forest.py
- try with 2 trees
- being able to run in ||
- being able to set diff
# ! include phonetics outside of the trie
TODO
"""
from typing import List, Tuple
from .trie import Trie

# ? How do we store multiple dictionaries and keep phonetic representations of them?
# ? How do we store multiple phonetic representations of the same dictionary?
# ? How do we store these within Tries?

# * Current idea would just be to create a bunch of dictionaries and a bunch
# * of tries...but that seems too memory intensive.
class Forest:
    def __init__(self):
        """
        TODO
        """
        self.tries = []

    def add_trie(self, trie: Trie):
        """
        TODO
        """
        self.tries.append(trie)

    def search(self, word: str) -> List[Tuple[str, float, float]]:
        """
        Conduct search with query word in parallel on all
        tries in the trie
        Args:
            word: query word
        Return:
            list of tuples that has potential patch with
                edit distance and jaro-winkler value
        """
        tentative_results = []

        # TODO get phonetic representation of input word
        # phonetic_word = phonetic_algorithm(word)

        # search each trie for the
        for i, trie in enumerate(self.tries):
            print(f"Searching Trie {i}")
            results = trie.search(word)
            print("Search results:", results)

        # ? Do we want to just return all results?
        # ? Or do we want some notion of "best" result?
        # ? Or do we want to return the best result for each trie?
        return tentative_results
