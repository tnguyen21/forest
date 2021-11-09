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

# * hacky way to add to python path -- need to find better way to mangage imports
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

    start_time = datetime.now()
    # load word list into trie
    print("Loading word list into trie...")
    with open("datasets/gazetteer_entries.txt", "r") as f:
        # TODO - this is pretty slow, can we find ways to speed it up?
        for line in f:
            word = line.strip()
            trie.add_entry(word)
    print("Done.")

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
    logging_data = {"query_results_len": {}}

    print("Performing queries...")
    for list_of_words in lists_of_words:
        # temporary place to store amount of results for given query
        edit_distance_data = {0: [], 1: [], 2: []}

        for word in list_of_words:
            for i in range(3):
                # ! there are some bugs when ED = 0
                # ! we should get exact matches but we dont
                print(f"query: {word}\nedit distance: {i}")
                results = trie.search(word, max_edit_distance=i)
                print(f"results: {results}")
                print("-----")
                edit_distance_data[i].append(len(results))

        # avg len of results for each edit distance
        for i in range(3):
            edit_distance_data[i] = sum(edit_distance_data[i]) / len(
                edit_distance_data[i]
            )
        logging_data["query_results_len"][len(list_of_words[0])] = edit_distance_data

    end_time = datetime.now()

    logging_data["runtime"] = end_time - start_time

    # log off results
    print("Logging results...")
    with open(f"word_len_experiment_run_{datetime.now().timestamp()}.pkl", "wb") as f:
        pickle.dump(logging_data, f)

    print("Done!")
