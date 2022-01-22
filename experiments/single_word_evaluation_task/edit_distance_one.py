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
import pickle
import datetime
import argparse
import json

logger_object = {}


def main():
    return 0


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

    main(args.trie_pkl_path, args.training_data_path)

    if args.save_data:
        with open(
            f"seriailzed_trie_size_run_{datetime.now().timestamp()}.json", "w"
        ) as f:
            json.dump(logger_object, f)
