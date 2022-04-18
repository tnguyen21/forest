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
        # ? can we have multiple phrases for the same concept id
        concept_id_entry = self.concept_id_expression_gazetteer.setdefault(concept_id, [])
        concept_id_entry = concept_id_entry.append(phrase)

        document = self.tokenizer(phrase)
        for token in document:
            # ? lowercase words before adding to word list
            # if token exists in dict, return the list corresponding to the entry
            # otherwise, initialize to empty list and return
            token_entry = self.word_expression_gazetteer.setdefault(token.text, [])
            token_entry = token_entry.append(phrase)


    def create_tries(self):
        """
        TODO
        """
        # ? api to set phonetic representation and other parameters to tries for the forest
        # ? do we just repeat params from phonetic_trie.py? not very DRY
        # create trie with no phonetic representation
        new_phonetic_trie = PhoneticTrie()
        new_phonetic_trie.add_trie()
        
        for word in self.word_expression_gazetteer.keys():
            new_phonetic_trie.add_entry(word)
        
        self.trie_list.append(new_phonetic_trie)

    def search(self, text: str) -> List[Dict]:
        """
        TODO
        """
        return []
