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
        "--data_path",
        help="Path to .txt file containing words",
        default=None,
        required=True,
    )

    args = parser.parse_args()

    main()

    with open(f"seriailzed_trie_size_run_{datetime.now().timestamp()}.json", "w") as f:
        json.dump(logger_object, f)
