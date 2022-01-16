""" phonetic_trie.py

Forest data structure which contains a list of Tries, each with different
hyperparameters (e.g. ED thresholds, phonetic representation, word length limits)

Provides interface to add tries and entries to the Forest, and conduct queries
across all Tries in the forest.

"""
from typing import Union, Callable, List, Tuple

from phonetics import metaphone, dmetaphone, soundex, nysiis
from Levenshtein import jaro_winkler, distance

from .trie import Trie


class PhoneticTrie:
    def __init__(self):
        self.tries = []
        self.phonetic_map = {}

    def add_trie(
        self,
        min_entry_len: int = -1,
        max_entry_len: int = 999,
        phonetic_representation: Union[Callable, None] = None,
        max_edit_distance: int = 2,
        min_jaro_winkler_sim: float = 0.0,
    ):
        """
        Add Trie to PhoneticTries with additional parameters.
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

    def search(self, word: str) -> List[Tuple[str, int, float]]:
        """
        Conduct search with query word in parallel on all
        tries in the trie

        * Note that words queried in the trie should be of length
        * [min_entry_len - ED, max_entry_len + ED]
        Args:
            word: query word
        Return:
            list of tuples that has potential matches of query with
                edit distance and jaro-winkler similiarity
        """
        # all of this is used for post processing at the end
        tentative_results = []
        original_word = word

        # search each trie for the word
        for t in self.tries:
            # convert word into phonetic representation (if any is needed)
            if t["phonetic_representation"] is not None:
                word = t["phonetic_representation"](word)

            # calculate if query is valid for trie based on entry len
            query_lower_bound = t["min_entry_len"] - t["trie"].max_edit_distance
            query_upper_bound = t["max_entry_len"] + t["trie"].max_edit_distance
            if len(word) >= query_lower_bound and len(word) <= query_upper_bound:
                results = t["trie"].search(word)

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
        # TODO move this into its own function?
        resulting_words = set()
        for results in tentative_results:
            for word, _, _ in results:
                if word not in resulting_words:
                    resulting_words.add(word)

        formatted_results = []

        for result_word in resulting_words:
            formatted_result = {}
            formatted_result[
                "original_word"
            ] = original_word  # todo rename to be more descrpitive
            formatted_result["result_word"] = result_word
            formatted_result["original_edit_distance"] = distance(
                original_word, result_word
            )
            formatted_result["original_jaro_winkler_similarity"] = round(
                jaro_winkler(original_word, result_word), 4
            )

            # double metaphone representation and metrics
            # * double metaphone returns two results -- we use the first
            dmetaphone_query = dmetaphone(original_word)[0]
            dmetaphone_result = dmetaphone(result_word)[0]
            formatted_result["dmetaphone_query"] = dmetaphone_query
            formatted_result["dmetaphone_result"] = dmetaphone_result
            formatted_result["dmetaphone_edit_distance"] = distance(
                dmetaphone_result, dmetaphone_query
            )
            formatted_result["dmetaphone_jaro_winkler_similarity"] = round(
                jaro_winkler(dmetaphone_result, dmetaphone_query), 4
            )

            # metaphone representation and metrics
            metaphone_query = metaphone(original_word)[0]
            metaphone_result = metaphone(result_word)[0]
            formatted_result["metaphone_query"] = metaphone_query
            formatted_result["metaphone_result"] = metaphone_result
            formatted_result["metaphone_edit_distance"] = distance(
                metaphone_result, metaphone_query
            )
            formatted_result["metaphone_jaro_winkler_similarity"] = round(
                jaro_winkler(metaphone_result, metaphone_query), 4
            )

            # soundex representation and metrics
            soundex_query = soundex(original_word)[0]
            soundex_result = soundex(result_word)[0]
            formatted_result["soundex_query"] = soundex_query
            formatted_result["soundex_result"] = soundex_result
            formatted_result["soundex_edit_distance"] = distance(
                soundex_result, soundex_query
            )
            formatted_result["soundex_jaro_winkler_similarity"] = round(
                jaro_winkler(soundex_result, soundex_query), 4
            )

            # nysiis representation and metrics
            nysiis_query = nysiis(original_word)[0]
            nysiis_result = nysiis(result_word)[0]
            formatted_result["nysiis_query"] = nysiis_query
            formatted_result["nysiis_result"] = nysiis_result
            formatted_result["nysiis_edit_distance"] = distance(
                nysiis_result, nysiis_query
            )
            formatted_result["nysiis_jaro_winkler_similarity"] = round(
                jaro_winkler(nysiis_result, nysiis_query), 4
            )

            formatted_results.append(formatted_result)

        # ? Or do we want some notion of "best" result?
        # ? Or do we want to return the best result for each trie?
        # TODO want to know where entry is coming from
        # ? figure out a way to weight similiarity scores
        # ? calc both sim scores between phonetic repr and original words?
        # * ultimately asking what are entries that are similar to my query
        # TODO should come up with some way to determine what we want returned from the search method
        # TODO i.e. don't always need to return ED, JW sim score, etc -- come up with an interface
        # TODO to allow users to specify what they want returned
        # * ---
        # TODO include length of original search word for each phonetic representation
        # TODO include original search word as well
        # TODO include len of phonetic representations
        # TODO include length of resulting word as well
        #! each word should appear in result once -- will be an object with all information someone would need
        # * ---
        # TODO would be nice to just see all similarity score for each phonetic representations we have in each trie
        return formatted_results
