""" experiments/multi_trie_experiment.py
This script runs a small experiment using the Princeton
Word Net database.

Observes the performance of the multi-trie search
by separating the dataset into two separate tries, one with
words of length 7 or less, and one with words of length
greater than 5. Search shorter words with ED=1, longer
words with ED=2.

Hoping to determine if multi-trie search with this method
runs faster than single-trie search, and by how much.
"""

# TODO hacky way to add to python path -- need to find better way to mangage imports
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # noqa: E501

from typing import Tuple, List
from trie import Trie
from datetime import datetime
import argparse
from pprint import pprint
import json

# Init data logging dict
# * not a great practice to make this a global object
# * but it's easier for all functions to log off timing and metrics
# * if it's a global object
data_logger = {}


def load_queries(query_file_path: str) -> Tuple[List, List]:
    """
    Given a path to the query file, return a list of
    queries, one for each of the length conditions
    considered in the experiment

    Args:
        query_file_path: path to the query file
    Return:
        A list of queries, one for each of the length
        conditions considered in the experiment
    """
    queries = []
    with open(query_file_path, "r") as f:
        for line in f:
            queries.append(line.strip())

    short_queries = [query for query in queries if len(query) <= 6]
    long_queries = [query for query in queries if len(query) >= 7]

    return (short_queries, long_queries)


def load_tries(dataset_file_path: str) -> Tuple[Trie, Trie, Trie]:
    """
    Given a path to the dataset, do some filtering
    on string lengths and add them to Tries. Return
    the Tries created for this experiment

    Args:
        dataset_file_path: path to the dataset
    Return:
        A tuple of three Tries, one for each of the
        length conditions considered in the experiment
    """
    trie_with_all_words = Trie()
    trie_with_long_words = Trie()
    trie_with_short_words = Trie()

    # * this is slow. adding entries to tries takes a long time
    with open(dataset_file_path, "r") as f:
        for word in f:
            word = word.strip()
            trie_with_all_words.add_entry(word)
            if len(word) <= 7:
                trie_with_short_words.add_entry(word)

            if len(word) >= 5:
                trie_with_long_words.add_entry(word)

    return (trie_with_all_words, trie_with_long_words, trie_with_short_words)


def one_trie_search(
    trie_with_all_words: Trie, short_queries: List, long_queries: List
) -> List:
    """
    Given a list of queries, conducts a search with all PWN words in one trie.
    Short words are searched with ED=1, long words are searched with ED=2.

    Args:
        trie_with_all_words: a trie containing all PWN words
        short_queries: a list of queries to search with ED=1
        long_queries: a list of queries to search with ED=2
    Return:
        A list of results for each query
    """
    query_results = []
    # ? do we want to log off timings for these searches
    for word in short_queries:
        results = trie_with_all_words.search(word, max_edit_distance=1)
        query_results.append(results)

    for word in long_queries:
        results = trie_with_all_words.search(word, max_edit_distance=2)
        query_results.append(results)

    return query_results


def multi_trie_search(
    trie_with_short_words: Trie,
    trie_with_long_words: Trie,
    short_queries: List,
    long_queries: List,
) -> List:
    """
    Given a list of queries, perform searches on two separate Tries. One trie
    contains words that have length 7 or less, the other contains words that
    have length greater than 5.

    Short words are searched with ED=1 in the "short words" Trie,
    long words are searched with ED=2 in the "long words" Trie.

    Args:
        trie_with_short_words: a trie containing words with length 7 or less
        trie_with_long_words: a trie containing words with length greater than 5
        short_queries: a list of queries to search with ED=1
        long_queries: a list of queries to search with ED=2
    Return:
        A list of results for each query

    """
    query_results = []

    for word in short_queries:
        results = trie_with_short_words.search(word, max_edit_distance=1)
        query_results.append(results)

    for word in long_queries:
        results = trie_with_long_words.search(word, max_edit_distance=2)
        query_results.append(results)

    return query_results


def main(queries_file_path: str, dataset_file_path: str):
    """
    Entry point for experiment. Loads in queries and dataset,
    conducts searches, and logs off results.
    """
    total_start_time = datetime.now()
    # Load queries
    print("Loading in queries..")
    query_loading_start_time = datetime.now()
    (short_queries, long_queries) = load_queries(queries_file_path)
    query_loading_end_time = datetime.now()

    data_logger["query_loading_time"] = float(
        query_loading_end_time.timestamp()
    ) - float(query_loading_start_time.timestamp())

    # Load data into tries
    print("Loading data into Tries...")
    trie_loading_start_time = datetime.now()
    (trie_with_all_words, trie_with_long_words, trie_with_short_words) = load_tries(
        dataset_file_path
    )
    trie_loading_end_time = datetime.now()
    data_logger["trie_loading_time"] = float(trie_loading_end_time.timestamp()) - float(
        trie_loading_start_time.timestamp()
    )

    # Run searches 3 times, average of 3 runs
    print("Running single Trie searches...")
    single_trie_search_times = []
    for _ in range(3):
        search_start_time = datetime.now()
        one_trie_results = one_trie_search(
            trie_with_all_words, short_queries, long_queries
        )
        search_end_time = datetime.now()
        single_trie_search_times.append(
            float(search_end_time.timestamp()) - float(search_start_time.timestamp())
        )

    data_logger["avg_one_trie_search_time"] = sum(single_trie_search_times) / len(
        single_trie_search_times
    )

    print("Running multi Trie searches...")
    multi_trie_search_times = []
    for _ in range(3):
        search_start_time = datetime.now()
        multi_trie_results = multi_trie_search(
            trie_with_short_words, trie_with_long_words, short_queries, long_queries
        )
        search_end_time = datetime.now()
        multi_trie_search_times.append(
            float(search_end_time.timestamp()) - float(search_start_time.timestamp())
        )

    data_logger["avg_multi_trie_search_time"] = sum(multi_trie_search_times) / len(
        multi_trie_search_times
    )
    print("Done!")
    total_end_time = datetime.now()

    data_logger["total_time"] = float(total_end_time.timestamp()) - float(
        total_start_time.timestamp()
    )
    # same searches, in same orders, so we expect them to produce
    # the same results
    for one_trie_results, multi_trie_results in zip(
        one_trie_results, multi_trie_results
    ):
        for one_trie_result, multi_trie_result in zip(
            one_trie_results, multi_trie_results
        ):
            assert one_trie_result == multi_trie_result

    # save data_logger dict into text
    with open(f"multi_trie_experiment_run_{datetime.now().timestamp()}.json", "w") as f:
        json.dump(data_logger, f)
    # ? multi-trie searches seem to be about 20% faster than single-trie searches
    # ? does this also apply at scale for very, very large tries?


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        help="Path to .txt file containing database of words",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--queries_path",
        help="Path to .txt file containing list of words we want to query",
        default=None,
        required=True,
    )
    args = parser.parse_args()

    main(args.queries_path, args.dataset_path)
