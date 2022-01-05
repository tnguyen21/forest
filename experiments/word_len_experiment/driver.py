""" word_len_experiment.py
This script runs a small experiment using the Princeton
Word Net database.

Words of varying length are loaded into a Trie, and then we
query words of varying lengths and various edit distances.

Results should approximately inform us of how many results
may turn up as a result of word length and edit distance. These
results can be a proxy in telling us how difficult it may be
to achieve performance in the NER task.

one liner to get words of specific length:
cat <txt_file> | grep -x '.\{<length>\}'
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
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="Path to .txt file containing words",
        default=None,
        required=True,
    )
    args = parser.parse_args()

    trie = Trie()

    start_time = datetime.now()
    # load word list into trie
    print("Loading word list into trie...")
    with open(args.data_path, "r") as f:
        # TODO - this is pretty slow, can we find ways to speed it up?
        for line in f:
            word = line.strip()
            trie.add_entry(word)
    print("Done.")

    # given that this object exists and is on local disk, we can load it
    # and save ~1.5 minutes of loading words
    # with open("experiments/word_len_experiment/pwn_trie_test.pkl", "rb") as f:
    #     trie = pickle.load(f)

    # NOTE these were hand picked from the word list to test
    four_letter_words = ["zeta", "yolk", "yogi", "wide", "vent", "user", "swim", "tame"]
    five_letter_words = [
        "zebra",
        "yodel",
        "wagon",
        "vowel",
        "towel",
        "tempo",
        "nerdy",
        "frown",
    ]
    six_letter_words = [
        "zigzag",
        "yonder",
        "wizard",
        "kettle",
        "iodize",
        "tender",
        "invert",
        "circus",
    ]
    seven_letter_words = [
        "zealous",
        "woodcut",
        "venture",
        "tribune",
        "sixfold",
        "mankind",
        "lexicon",
        "hurried",
    ]
    eight_letter_words = [
        "ziggurat",
        "youthful",
        "ultimate",
        "tyrannic",
        "isolated",
        "highness",
        "glumness",
        "devilish",
    ]
    nine_letter_words = [
        "zoologist",
        "youngster",
        "authentic",
        "enchanted",
        "hypocrisy",
        "iconology",
        "insurance",
        "memorable",
    ]

    lists_of_words = [
        four_letter_words,
        five_letter_words,
        six_letter_words,
        seven_letter_words,
        eight_letter_words,
        nine_letter_words,
    ]

    # store some data for query results
    logging_data = {"query_results_metrics": {}}

    print("Performing queries...")
    for list_of_words in lists_of_words:
        print(f"Querying words that are of len: {len(list_of_words[0])}")
        # temporary place to store amount of results for given query
        edit_distance_data = {
            0: {"avg_query_results_len": [], "avg_run_time": []},
            1: {"avg_query_results_len": [], "avg_run_time": []},
            2: {"avg_query_results_len": [], "avg_run_time": []},
        }

        for word in list_of_words:
            for i in range(3):
                # print(f"query: {word}\nedit distance: {i}")
                search_start_time = datetime.now()
                results = trie.search(word, max_edit_distance=i)
                search_end_time = datetime.now()
                # print(f"results: {results}")
                # print("-----")
                edit_distance_data[i]["avg_query_results_len"].append(len(results))
                edit_distance_data[i]["avg_run_time"].append(
                    (search_end_time - search_start_time).total_seconds()
                )

        for i in range(3):
            # avg len of results for each edit distance
            edit_distance_data[i]["avg_query_results_len"] = sum(
                edit_distance_data[i]["avg_query_results_len"]
            ) / len(edit_distance_data[i]["avg_query_results_len"])

            # avg run time for each edit distance
            edit_distance_data[i]["avg_run_time"] = sum(
                edit_distance_data[i]["avg_run_time"]
            ) / len(edit_distance_data[i]["avg_run_time"])
        logging_data["query_results_metrics"][
            len(list_of_words[0])
        ] = edit_distance_data

    end_time = datetime.now()

    logging_data["runtime"] = (end_time - start_time).total_seconds()

    # log off results
    print("Logging results...")
    with open(f"word_len_experiment_run_{datetime.now().timestamp()}.json", "w") as f:
        json.dump(logging_data, f)

    print("Done!")
