"""
TODO
"""

# TODO hacky way to add to python path -- need to find better way to mangage imports
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # noqa: E501


from trie import Trie
from common import load_trie_from_pkl
from datetime import datetime
import pandas as pd
import argparse
import json

logger_object = {"avg_search_time_per_word": 0, "false_postive_count": 0}


def main(trie_pkl_path: str, train_data_path: str):
    """
    TODO

    Args:
        trie_pkl_path: TODO
        train_data_path: TODO
    """
    # Load trie
    trie = load_trie_from_pkl(trie_pkl_path)

    # Set max edit distance of trie to 0 for exact matches
    trie.max_edit_distance = 2

    # Load data
    data_df = pd.read_csv(train_data_path)

    # Keep track of amount of time it takes to search for each word
    exact_match_search_times = []

    for _, query, expected_ed in data_df.itertuples(index=False):
        start_time = datetime.now()
        results = trie.search(query)
        end_time = datetime.now()
        exact_match_search_times.append((end_time - start_time).total_seconds())
        print((end_time - start_time).total_seconds())

    # Calculate average search time
    logger_object["avg_search_time_per_word"] = sum(exact_match_search_times) / len(
        exact_match_search_times
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trie_pkl_path",
        help="Path to .pkl file containing trie loaded with all words",
        default=None,
        required=True,
    )

    parser.add_argument(
        "--training_data_path",
        help="Path to .csv file containing training data",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--save_data",
        help="Boolean flag to log out .json file containing results",
        default=False,
        required=False,
    )

    args = parser.parse_args()

    script_start_time = datetime.now()
    main(args.trie_pkl_path, args.training_data_path)
    script_end_time = datetime.now()

    logger_object["total_runtime"] = (
        script_end_time - script_start_time
    ).total_seconds()

    # Print out results
    print(json.dumps(logger_object))

    if args.save_data:
        with open(
            f"seriailzed_trie_size_run_{datetime.now().timestamp()}.json", "w"
        ) as f:
            json.dump(logger_object, f)
