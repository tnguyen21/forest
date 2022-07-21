""" forest.py
Implementation of Forest class composed of multiple Tries to conduct
multi-word expression search.
"""

import spacy

from typing import List, Dict
from .phonetic_trie import PhoneticTrie
from sklearn.linear_model import LogisticRegression
from pprint import pprint


class Forest:
    def __init__(
        self,
        logistic_regression_model: LogisticRegression = None,
        logistic_regression_threshold: float = 0.5,
    ):
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
        self.word_expression_gazetteer = {}  # generated from expression
        self.word_concept_id_gazetteer = {}  # generated from expression
        self.expression_count = 0
        self.concept_id_count = 0
        self.word_to_word_determining_score = {}
        self.word_to_cuid_determining_score = {}
        self.logistic_regression_model = logistic_regression_model
        self.logistic_regression_model_threshold = logistic_regression_threshold

        # https://spacy.io/usage/spacy-101#annotations-token
        self.tokenizer = spacy.load("en_core_web_sm")

    def set_logistic_regression_model(
        self, logistic_regression_model: LogisticRegression
    ):
        """
        Add logistic regression  model to Forest

        args:
            logistic_regression_model: sklearn.linear_model.LogisticRegression
        """
        self.logistic_regression_model = logistic_regression_model

    def add_phrase(self, concept_id: str, phrase: str) -> None:
        """
        Add phrase and corresponding concept ID to gazetteers maintained by Forest

        args:
            concept_id: concept ID (CUID) corresponding to phrase
            phrase: expression within dictionary we would like Forest to identify

        """
        # ? should we lowercase the phrase before adding it to the gazetteer
        # ? can we have multiple phrases for the same concept id
        concept_id_entry = self.concept_id_expression_gazetteer.setdefault(
            concept_id, []
        )
        concept_id_entry = concept_id_entry.append(phrase)

        # increment expression count
        self.expression_count += 1

        document = self.tokenizer(phrase)
        for token in document:
            # ? lowercase words before adding to word list
            # if token exists in dict, return the list corresponding to the entry
            # otherwise, initialize to empty list
            word_expression_token_entry = self.word_expression_gazetteer.setdefault(
                token.text, []
            )
            word_expression_token_entry = word_expression_token_entry.append(phrase)

            word_cuid_token_entry = self.word_concept_id_gazetteer.setdefault(
                token.text, []
            )
            word_cuid_token_entry = word_cuid_token_entry.append(concept_id)

    def create_tries(self):
        """
        Create PhoneticTrie and add to list of PhoneticTries
        used by Forest for fuzzy string search
        """
        # ? api to set phonetic representation and other parameters to tries for the forest
        # ? do we just repeat params from phonetic_trie.py? not very DRY
        # create trie with no phonetic representation
        new_phonetic_trie = PhoneticTrie()
        new_phonetic_trie.add_trie()

        for word in self.word_expression_gazetteer.keys():
            new_phonetic_trie.add_entry(word)

        self.phonetic_trie_list.append(new_phonetic_trie)

    def calculate_determining_scores(self) -> None:
        """
        Calculate "determining scores" for every word that appears in the expression
        gazetteer. "Determining score" is a fraction roughly corresponding to how
        unique that word is for determining an expression within our dictionary.

        additional notes:
            should be called after all expressions have been added to the Forest.
            for every phrase added after
        """
        self.concept_id_count = len(self.concept_id_expression_gazetteer.keys())

        for word, expression_list in self.word_expression_gazetteer.items():
            # if only one expression word occurs in then score = 1
            self.word_to_word_determining_score[word] = (
                self.expression_count + 1 - len(expression_list)
            ) / self.expression_count

        for word, cuid_list in self.word_concept_id_gazetteer.items():
            # if word only has one associated CUID, then score = 1
            self.word_to_cuid_determining_score[word] = (
                self.concept_id_count + 1 - len(cuid_list)
            ) / self.concept_id_count

        # ? do multiple expressions have the same CUID in dsyn

    def get_token_concept_dictionary(self, text: str) -> Dict[str, List[Dict]]:
        """
        Given non-tokenized block of text, tokenize text, then fuzzy-search on every word
        to identify possible matches.
        """

        # split text on tokens
        document = self.tokenizer(text)

        # Dict[String: Tuple[Expr, CUID, Expr Len, Token Len, Token Position,
        #                    Expr Det Score, CUID Det Score, Word Distance]]
        token_concept_dictionary = {}

        for token in document:
            # search in tries
            related_concepts = []
            for trie in self.phonetic_trie_list:
                results = trie.search(
                    word=token.text,
                    weight_score_threshold=0.9,  # arbitrary threshold, just set high enough to reduce amount of results
                )
                # find corresponding concept(s) for each result
                for result in results:
                    result_word = result["result"]
                    related_expressions = self.word_expression_gazetteer.get(
                        result_word, None
                    )
                    # print("result_word:", result_word, "-- related expressions:", related_expressions)
                    # ! this is really slow and bad, but just to get this working
                    # ! this to inefficient looping
                    for expression in related_expressions:
                        for (
                            concept_id,
                            expressions,
                        ) in self.concept_id_expression_gazetteer.items():
                            if expression in expressions:
                                related_concepts.append(
                                    (
                                        result_word,  # can be word that is similar to token in text as well
                                        expression,
                                        concept_id,
                                        len(expression.split(" ")),
                                        len(result_word),
                                        expression.find(
                                            result_word
                                        ),  # token position in expression
                                        self.word_to_word_determining_score.get(
                                            result_word, 0
                                        ),
                                        self.word_to_cuid_determining_score.get(
                                            result_word, 0
                                        ),
                                    )
                                )
            # map token back to concept ids
            # ? this is the token text -- could have multiple of the same text
            # ? in different positions...are tokens hashable?
            token_concept_dictionary[token.text] = related_concepts
        # need to figure out for each token in input whether or not that token is the
        # start of an expression, and if so, which expression
        return token_concept_dictionary

    def format_results(self, search_text: str, search_results: Dict[str, List[Dict]]) -> List[List]:
        # header row
        formatted_results = ["Input", "Searched Token", "Matched Token", "Concept ID", "Concept Name"]

        for token, matches in search_results.items():
            for match in matches:
                matched_token = match[0]
                match_concept_name = match[1]
                match_cuid = match[2]
                formatted_row = [
                    search_text,
                    token,
                    matched_token,
                    match_cuid,
                    match_concept_name
                ]

                formatted_results.append(formatted_row)

        return formatted_results

    def search(self, text: str, use_lr_model: bool = False) -> List[Dict]:
        """
        
        args:
            text: sentence to identify named-entities within
        """
        token_concept_dictionary = self.get_token_concept_dictionary(text)
        
        # for token, matched_concepts in token_concept_dictionary.items():
        #     print(token, matched_concepts)
        #     break
            
        if (use_lr_model):
            # use LR model to filter our concepts from search result
            # in token-concept dictionary match
            print('use logistic regression model for search')
        else:
            # some logic for filtering out potential matches
            print('do some logic here')

        formatted_return = self.format_results(text, token_concept_dictionary)
        
        return formatted_return
