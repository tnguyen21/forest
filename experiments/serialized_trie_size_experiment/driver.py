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
from datetime import datetime
import pickle
import argparse
import json

logger_object = {}


def main(text_data_path: str, trie_output_path: str):
    trie = Trie()

    print("Loading word list into trie...")
    # time how long it takes to read the input file
    data_read_start_time = datetime.now()
    with open(text_data_path, "r") as f:
        for line in f:
            word = line.strip()
            trie.add_entry(word)
    data_read_end_time = datetime.now()
    print("Done.")
    logger_object["input_reading_timing"] = str(
        data_read_end_time - data_read_start_time
    )

    # log how many words input into trie
    num_lines = sum(1 for line in open(text_data_path))
    print(f"Read {num_lines} lines.")
    logger_object["num_lines"] = num_lines

    # pickle trie object
    pickle_start_time = datetime.now()
    with open(trie_output_path, "wb") as f:
        pickle.dump(trie, f)
    pickle_end_time = datetime.now()
    logger_object["pkl_out_timing"] = str(pickle_end_time - pickle_start_time)

    # log how long it takes to read pkl file in
    pkl_read_start_time = datetime.now()
    with open(trie_output_path, "rb") as f:
        trie = pickle.load(f)
    pkl_read_end_time = datetime.now()
    logger_object["pkl_in_timing"] = str(pkl_read_end_time - pkl_read_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="Path to .txt file containing words",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--trie_output_path",
        help="Path to serialized trie output file",
        default=None,
        required=True,
    )
    args = parser.parse_args()

    main(text_data_path=args.data_path, trie_output_path=args.trie_output_path)

    with open(f"seriailzed_trie_size_run_{datetime.now().timestamp()}.json", "w") as f:
        json.dump(logger_object, f)
