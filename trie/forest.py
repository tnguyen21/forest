""" forest.py
TODO
"""

import spacy

from typing import List, Dict
from .phonetic_trie import PhoneticTrie
from pprint import pprint

class Forest:
    def __init__(self):
        """
        args:
            TODO info on each argument
        
        additional notes:
            concept_id_expression_gazetteer: Dict[str, list[str]]
            word_expression_gazetteer: Dict[str, list[str]]
            word_to_word_determining_score = Dict[str, float]
        """
        self.phonetic_trie_list = []
        self.concept_id_expression_gazetteer = {}
        self.word_expression_gazetteer = {} # generated from phrases? 
        self.word_to_word_determining_score = {}

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
        
        self.phonetic_trie_list.append(new_phonetic_trie)

    def search(self, text: str) -> List[Dict]:
        """
        TODO
        """

        # split text on tokens
        document = self.tokenizer(text)

        # Dict[String: Tuple[Expr, CUID, Expr Len, Token Len, Valid Tokens, 
        #                    Expr Det Score, CUID Det Score, Word Distance]]
        token_concept_dictionary = {}

        for token in document:
            # search in tries
            concept_ids = []
            for trie in self.phonetic_trie_list:
                results = trie.search(token.text)
                # find corresponding concept(s) for each result
                for result in results:
                    result_word = result["result"]
                    related_concepts = []
                    related_expressions = self.word_expression_gazetteer.get(result_word, None)
                    # print("result_word:", result_word, "-- related expressions:", related_expressions)
                    # ! this is really slow and bad, but just to get this working
                    # ! this to inefficient looping
                    for expression in related_expressions:
                        for concept_id, expressions in self.concept_id_expression_gazetteer.items():
                            if expression in expressions:
                                related_concepts.append(concept_id)
                    
                    if related_concepts is not None:
                        concept_ids = concept_ids + related_concepts

            # map token back to concept ids
            # ? this is the token text -- could have multiple of the same text
            # ? in different positions...are tokens hashable?
            token_concept_dictionary[token.text] = concept_ids 
            
        pprint(token_concept_dictionary)
        return []
