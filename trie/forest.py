""" forest.py
TODO
"""

from typing import List, Dict
from .phonetic_trie import PhoneticTrie

class Forest:
    def __init__(self):
        """
        args:
            TODO info on each argument
        
        additional notes:
            expression_dictionary: Dict[str, list[str]]
        """
        self.trie_list = []
        self.expression_dictionary = {"ID": "phrase"}
        self.word_list = [] # generated from phrases?

    def add_phrase(self, phrase: str) -> None:
        """
        TODO
        """
        return 0

    def add_trie(self, trie: PhoneticTrie):
        """
        TODO
        """
        return 0

    def search(self, text: str) -> List[Dict]:
        """
        TODO
        """
        return []
