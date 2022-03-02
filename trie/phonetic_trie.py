""" phonetic_trie.py

Forest data structure which contains a list of Tries, each with different
hyperparameters (e.g. ED thresholds, phonetic representation, word length limits)

Provides interface to add tries and entries to the Forest, and conduct queries
across all Tries in the forest.

"""
from typing import Union, Callable, List, Tuple, Dict

from phonetics import metaphone, dmetaphone
from jellyfish import soundex
from fuzzy import nysiis
from Levenshtein import jaro_winkler, distance
from sklearn.linear_model import LogisticRegression
import pandas as pd

from .trie import Trie


class PhoneticTrie:
    def __init__(
        self,
        logistic_regression_model: LogisticRegression = None,
        logistic_regression_threshold: float = 0.5,
    ):
        self.tries = []
        self.phonetic_map = {}
        self.logistic_regression_model = logistic_regression_model
        self.logistic_regression_model_threshold = logistic_regression_threshold

    def set_logistic_regression_model(
        self, logistic_regression_model: LogisticRegression
    ):
        """
        Add logistic regression model to PhoneticTrie

        Args:
            logistic_regression_model: logistic regression model
        """
        self.logistic_regression_model = logistic_regression_model

    def add_trie(
        self,
        trie: Trie = None,
        min_entry_len: int = -1,
        max_entry_len: int = 999,
        phonetic_representation: Union[Callable, None] = None,
        max_edit_distance: int = 2,
        min_jaro_winkler_sim: float = 0.0,
    ):
        """
        Add Trie to PhoneticTries with additional parameters.
        Args:
            min_entry_len: minimum length of entry to be added to Trie
            max_entry_len: maximum length of entry to be added to Trie
            phonetic_representation: phonetic representation of entries in Trie
            max_edit_distance: maximum edit distance for result
            min_jaro_winkler_sim: minimum jaro winkler similarity for search result
        """
        if trie:
            t = trie
        else:
            t = Trie(max_edit_distance, min_jaro_winkler_sim)
        trie_data = {
            "trie": t,
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
                # convert entry into phonetic representation (if any is needed)
                # and add to phonetic map
                if t["phonetic_representation"] is not None:
                    # check if phonetic function is in our map
                    if t["phonetic_representation"] not in self.phonetic_map:
                        self.phonetic_map[t["phonetic_representation"]] = {}

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
                # if there is no phonetic representation, just add entry
                else:
                    t["trie"].add_entry(entry)

    def format_results(
        self,
        query_word: str,
        results_list: List[str],
        metaphone_output: bool = False,
        dmetaphone_output: bool = False,
        soundex_output: bool = False,
        nysiis_output: bool = False,
    ) -> List[Dict]:
        """
        Helper function which formats a single search result such that
        a user can specify what they want to see in the result.

        Always returns calculated edit distance and jaro-winkler similarity.

        By default returns only queried word & result from searching in trie.

        Args:
            results_list: word from search result
            metaphone_output: if True, returns metaphone output
            dmetaphone_output: if True, returns dmetaphone output
            soundex_output: if True, returns soundex output
            nysiis_output: if True, returns nysiis output
        Return:
            List[Dict] of formatted results
        """
        formatted_results = []
        for result_word in results_list:
            formatted_result = {}

            formatted_result["query"] = query_word
            formatted_result["result"] = result_word
            formatted_result["original_edit_distance"] = distance(
                query_word, result_word
            )
            formatted_result["original_jaro_winkler_similarity"] = round(
                jaro_winkler(query_word, result_word), 4
            )

            if metaphone_output:
                metaphone_query = metaphone(query_word)
                metaphone_result = metaphone(result_word)
                formatted_result["metaphone_query"] = metaphone_query
                formatted_result["metaphone_result"] = metaphone_result
                formatted_result["metaphone_edit_distance"] = distance(
                    metaphone_result, metaphone_query
                )
                formatted_result["metaphone_jaro_winkler_similarity"] = round(
                    jaro_winkler(metaphone_result, metaphone_query), 4
                )

            if dmetaphone_output:
                # * double metaphone returns two results -- we use the first
                dmetaphone_query = dmetaphone(query_word)[0]
                dmetaphone_result = dmetaphone(result_word)[0]
                formatted_result["dmetaphone_query"] = dmetaphone_query
                formatted_result["dmetaphone_result"] = dmetaphone_result
                formatted_result["dmetaphone_edit_distance"] = distance(
                    dmetaphone_result, dmetaphone_query
                )
                formatted_result["dmetaphone_jaro_winkler_similarity"] = round(
                    jaro_winkler(dmetaphone_result, dmetaphone_query), 4
                )

            if soundex_output:
                soundex_query = soundex(query_word)
                soundex_result = soundex(result_word)
                formatted_result["soundex_query"] = soundex_query
                formatted_result["soundex_result"] = soundex_result
                formatted_result["soundex_edit_distance"] = distance(
                    soundex_result, soundex_query
                )
                formatted_result["soundex_jaro_winkler_similarity"] = round(
                    jaro_winkler(soundex_result, soundex_query), 4
                )

            if nysiis_output:
                nysiis_query = nysiis(query_word)
                nysiis_result = nysiis(result_word)
                formatted_result["nysiis_query"] = nysiis_query
                formatted_result["nysiis_result"] = nysiis_result
                formatted_result["nysiis_edit_distance"] = distance(
                    nysiis_result, nysiis_query
                )
                formatted_result["nysiis_jaro_winkler_similarity"] = round(
                    jaro_winkler(nysiis_result, nysiis_query), 4
                )

            formatted_results.append(formatted_result)

        return formatted_results

    def filter_results(
        self,
        query_word: str,
        results_list: List[str],
        metaphone_weight: float = 1,
        dmetaphone_weight: float = 1,
        soundex_weight: float = 1,
        nysiis_weight: float = 1,
        weight_score_threshold: float = 0.0,
    ) -> List[str]:
        """
        Helper function which filters results based on phonetic weights.
        Args:
            results_list: list of results from search
            metaphone_weight: weight for jw sim score for metaphone reprensentation
            dmetaphone_weight: weight for jw sim score for dmetaphone reprensentation
            soundex_weight: weight for jw sim score for soundex reprensentation
            nysiis_weight: weight for jw sim score for nysiis reprensentation
            weight_score_threshold: threshold for weighted score
        Return:
            List of filtered results
        """
        filtered_results = []

        for result in results_list:

            if self.logistic_regression_model is not None:
                # if LR model is loaded, use it to predict probability
                # of result being a match for query and filter accordingly

                # LR model takes in vector of EDs and JW sims, calculate them all here
                # TODO this comptuation is repeated in format_results -- refactor
                original_edit_distance = distance(query_word, result)
                original_jaro_winkler_similarity = round(
                    jaro_winkler(query_word, result), 4
                )

                # * double metaphone returns two results -- we use the first
                dmetaphone_query = dmetaphone(query_word)[0]
                dmetaphone_result = dmetaphone(result)[0]
                dmetaphone_edit_distance = distance(dmetaphone_result, dmetaphone_query)
                dmetaphone_jaro_winkler_similarity = round(
                    jaro_winkler(dmetaphone_result, dmetaphone_query), 4
                )

                metaphone_query = metaphone(query_word)
                metaphone_result = metaphone(result)
                metaphone_edit_distance = distance(metaphone_result, metaphone_query)
                metaphone_jaro_winkler_similarity = round(
                    jaro_winkler(metaphone_result, metaphone_query), 4
                )

                soundex_query = soundex(query_word)
                soundex_result = soundex(result)
                soundex_edit_distance = distance(soundex_result, soundex_query)
                soundex_jaro_winkler_similarity = round(
                    jaro_winkler(soundex_result, soundex_query), 4
                )

                nysiis_query = nysiis(query_word)
                nysiis_result = nysiis(result)
                nysiis_edit_distance = distance(nysiis_result, nysiis_query)
                nysiis_jaro_winkler_similarity = round(
                    jaro_winkler(nysiis_result, nysiis_query), 4
                )

                # lots of overhead in creating this array to do an inference
                X = pd.DataFrame(
                    [
                        [
                            dmetaphone_edit_distance,
                            dmetaphone_jaro_winkler_similarity,
                            metaphone_edit_distance,
                            metaphone_jaro_winkler_similarity,
                            soundex_edit_distance,
                            soundex_jaro_winkler_similarity,
                            nysiis_edit_distance,
                            nysiis_jaro_winkler_similarity,
                            original_edit_distance,
                            original_jaro_winkler_similarity,
                        ]
                    ],
                    columns=[
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
                    ],
                )

                # predict probability of result being a match for query
                # and filter accordingly
                proba = self.logistic_regression_model.predict_proba(X)
                if proba[0][1] > self.logistic_regression_model_threshold:
                    filtered_results.append(result)

            else:
                # use simple threshold to filter results
                # calculate weighted score
                metaphone_score = jaro_winkler(metaphone(query_word), metaphone(result))

                # double metaphone returns two results -- we use the first one
                dmetaphone_score = jaro_winkler(
                    dmetaphone(query_word)[0], dmetaphone(result)[0]
                )

                soundex_score = jaro_winkler(soundex(query_word), soundex(result))

                nysiis_score = jaro_winkler(nysiis(query_word), nysiis(result))

                scores = [
                    metaphone_score,
                    dmetaphone_score,
                    soundex_score,
                    nysiis_score,
                ]

                weighted_score = sum(scores) / len(scores)

                if weighted_score >= weight_score_threshold:
                    filtered_results.append(result)

        return filtered_results

    def search(
        self,
        word: str,
        max_edit_distance: int = 2,
        # ? pass in weights as dictionaries
        # TODO move this to class variables, and have set_weights functions
        metaphone_weight: float = 1,
        dmetaphone_weight: float = 1,
        soundex_weight: float = 1,
        nysiis_weight: float = 1,
        # TODO move this to class variables
        weight_score_threshold: float = 0.0,
        # TODO move these to class variables, create set_outputs function
        # * only want to set outputs once and not have to remember for every search
        metaphone_output: bool = False,
        dmetaphone_output: bool = False,
        soundex_output: bool = False,
        nysiis_output: bool = False,
    ) -> Dict:
        """
        Conduct search with query word in parallel on all
        tries in the trie

        * Note that words queried in the trie should be of length
        * [min_entry_len - ED, max_entry_len + ED]
        Args:
            word: query word
            metaphone_weight: weight for jw sim score for metaphone reprensentation
            dmetaphone_weight: weight for jw sim score for dmetaphone reprensentation
            soundex_weight: weight for jw sim score for soundex reprensentation
            nysiis_weight: weight for jw sim score for nysiis reprensentation
            weight_score_threshold: threshold for weighted score
            metaphone_output: whether to output metaphone results
            dmetaphone_output: whether to output dmetaphone results
            soundex_output: whether to output soundex results
            nysiis_output: whether to output nysiis results
        Return:
            dict of formatted results
        """
        # all of this is used for post processing at the end
        tentative_results = []
        original_word = word
        #! if we are using multiple tries w phonetic representations
        #! cannot filter results in forest -- should return all the results
        #! OR if we have a threshold passed in
        # search each trie for the word
        for t in self.tries:
            # convert word into phonetic representation (if any is needed)
            if t["phonetic_representation"] is not None:
                word = t["phonetic_representation"](word)

            # calculate if query is valid for trie based on entry len
            query_lower_bound = t["min_entry_len"] - t["trie"].max_edit_distance
            query_upper_bound = t["max_entry_len"] + t["trie"].max_edit_distance
            if len(word) >= query_lower_bound and len(word) <= query_upper_bound:
                results = t["trie"].search(word, max_edit_distance)

                # if phonetic representation exists, need to do post-processing
                if t["phonetic_representation"] is not None:
                    mapped_words = [
                        (
                            self.phonetic_map[t["phonetic_representation"]][word],
                            ed,
                            jw_sim,
                        )
                        for word, ed, jw_sim in results
                    ]
                    mapped_results = []
                    for words, ed, jw_sim in mapped_words:
                        for word in words:
                            mapped_results.append((word, ed, jw_sim))
                    tentative_results.append(mapped_results)
                # if phonetic representation does not exist, just add results
                else:
                    tentative_results.append(results)

        # more post processing -- it's at this point we will lose the
        # association of a result to a particular trie
        resulting_words = set()
        for results in tentative_results:
            for word, _, _ in results:
                if word not in resulting_words:
                    resulting_words.add(word)

        if len(resulting_words) == 0:
            # if no results returned, return empty search term
            resulting_words.add("")

        filtered_results = self.filter_results(
            original_word,
            resulting_words,
            metaphone_weight,
            dmetaphone_weight,
            soundex_weight,
            nysiis_weight,
            weight_score_threshold,
        )

        formatted_results = self.format_results(
            original_word,
            filtered_results,
            metaphone_output,
            dmetaphone_output,
            soundex_output,
            nysiis_output,
        )

        return formatted_results
