""" forest.py
TODO
"""

import spacy

from typing import List, Dict
from .phonetic_trie import PhoneticTrie

class Forest:
    def __init__(self):
        """
        args:
            TODO info on each argument
        
        additional notes:
            expression_gazetteer: Dict[str, list[str]]
            expression_gazetteer: Dict[str, list[str]]
        """
        self.trie_list = []
        self.concept_id_expression_gazetteer = {}
        self.word_expression_gazetteer = {} # generated from phrases? 
        
        # https://spacy.io/usage/spacy-101#annotations-token
        self.tokenizer = spacy.load("en_core_web_sm")

    def add_phrase(self, concept_id: str, phrase: str) -> None:
        """
        TODO
        """
        # ? should we lowercase the phrase before adding it to the gazetteer
        self.concept_id_expression_gazetteer[concept_id] = phrase

        document = self.tokenizer(phrase)
        for token in document:
            # ? lowercase words before adding to word list
            # if token exists in dict, return the list corresponding to the entry
            # otherwise, initialize to empty list and return
            token_entry = self.word_expression_gazetteer.setdefault(token.text, [])
            token_entry = token_entry.append(phrase)


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
