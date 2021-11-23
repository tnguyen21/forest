""" experiments/multi_trie_experiment.py
This script runs a small experiment using the Princeton
Word Net database.

TODO comment

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

if __name__ == "__main__":
    trie = Trie()
