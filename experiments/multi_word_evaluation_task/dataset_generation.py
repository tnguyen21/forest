""" dataset_generation.py

Code which generates evaluation dataset for multi-word matching task.

"""

# TODO hacky way of adding packages to path; find better way of doing this

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # noqa: E501

from trie import Forest
from typing import Set, List
from tqdm import tqdm
from pprint import pprint
import random
import csv
import json
import dill as pickle
import pandas as pd


def create_forest(dictionary_path: str, output_forest_path: str):
    """
    Loads input dictionary of expressions into Forest then pickles
    Forest to `output_forest_path`
    
    args:
        dictionary_path: path to csv containing dictionary with concept IDs and expressions
        output_forest_path: path to output serialized Forest
    """
    forest = Forest()
    
    dict_df = pd.read_csv(dictionary_path)

    for _, id, expression in tqdm(dict_df.itertuples(), desc="Load dictionary into Forest..."):
        forest.add_phrase(id, expression)

    forest.create_tries()
    forest.calculate_determining_scores()

    with open(output_forest_path, "wb") as f:
        pickle.dump(forest, f)
    

def add_stop_words(dictionary_words: List[str], sentence: List[str]) -> List[str]:
    """
    add [0, 5] filler words between expressions in sentence
    where filler words is a random choice from: prepositions/stop words,
    1000 randomly chosen words from PWN, and a set of words from the input
    dict

    args:
        dictionary_words: set of individual words from terms in dict
        sentence: constructed sentence with annotations for training
    """
    # TODO move this out of this function eventually to prevent multiple reading-ins
    pwn_stop_words_path = "datasets/1000_gazetteer_entries.txt"
    with open(pwn_stop_words_path, "r") as f:
        pwn_stop_words = [word.strip() for word in f.readlines()]

    # ? include all prepositions
    stop_words = ["a", "and", "the", "without", "from", "to", "since", "until"]

    filler_words = stop_words + pwn_stop_words + dictionary_words

    # add [0, 5] filler words between expressions in sentence
    for _ in range(random.randint(0, 5)):
        sentence += [random.choice(filler_words)]

    return sentence


def generate_sample_sentences(dictionary_path: str, dataset_output_path: str) -> None:
    """
    given path to dictionary with cuid and term,
    generate a dataset to use for the multi-word matching task.
    outputs dictionary as a csv
    """
    training_data = []
    # read csv into List[Tuple[str, str]]
    with open(dictionary_path, "r") as f:
        reader = csv.reader(f)
        dictionary = [tuple(row) for row in list(reader)]

	# set up file writer for csv file
    # output_data_file = open(dataset_output_path, "w")
    # output_csv_writer = csv.writer(output_data_file, delimiter=",")

    # create set of unique words from dictionary
    dictionary_terms_set = set()
    for (_, entry) in dictionary:
        # assume words in entry are separated by space
        for word in entry.split(" "):
            dictionary_terms_set.add(word)

	# convert set->list for ease of use later
    dictionary_terms = list(dictionary_terms_set)

    sample_sentences = {}

    for (cuid, entry) in tqdm(dictionary, desc="Generating sample sentences..."):
        # randomly choose how many sentences will start with entry
        number_of_sentences_that_start_with_entry = random.randint(1, 3)
        # ? token or character position -- token position
        # ? separating cuids -- don't mix expressions with same CUID
        for _ in range(number_of_sentences_that_start_with_entry):
            # randomly choose [1, 5] expressions in sentence
            number_of_additional_expressions = random.randint(0, 5)

            # begin sentence with entry
            sentence = entry.split(" ")
            
            # save each label for annoations as a tuple of (int, int, str)
            # where it is (start token idx, end token idx, expression)
            annotations = []
                   
            annotations.append([0, len(sentence) - 1, entry])

            # add additional expressions to sentence
            for _ in range(number_of_additional_expressions):
                sentence = add_stop_words(dictionary_terms, sentence=sentence)

                # add another expression to the sentence, 2nd element in tuple
                new_expression = random.choice(dictionary)[1]
                start_token_idx = len(sentence) - 1
                sentence += new_expression.split(" ")
                end_token_idx = len(sentence) - 1
                annotations.append([start_token_idx, end_token_idx, new_expression])
            
            # write out sentence and annotations to dataset
            joined_sentence = " ".join(sentence)
            sample_sentences[joined_sentence] = []
            for annotation in annotations:
                sample_sentences[joined_sentence].append(annotation)
    
    with open(dataset_output_path, "w") as f:
        json.dump(sample_sentences, f, indent=2)
    
	# remember to close file after done writing!
    # output_data_file.close()

def generate_dataset(
    sample_sentence_dataset_path: str,
    forest_pkl_path: str,
    output_dataset_path: str
):
    """
    Generate dataset to use as input for training logistic regression
    model for recognizing the beginning of multi-word expressions
    """
    # loading in dataset and serialized fores
    with open(sample_sentence_dataset_path, "r") as f:
        sample_sentence_dict = json.loads(f.read())
    
    forest = None
    with open(forest_pkl_path, "rb") as f:
        forest = pickle.load(f)

    pprint(sample_sentence_dict)
    


if __name__ == "__main__":
    dictionary_input_path = "datasets/nasa_shared_task/HEXTRATO_dictionary.csv"
    dataset_output_path = "experiments/multi_word_evaluation_task/test.json" 
    forest_output_path = "test_forest.pkl"
    
    create_forest(dictionary_input_path, forest_output_path)
    generate_sample_sentences(dictionary_input_path, dataset_output_path)
    generate_dataset(dataset_output_path, forest_output_path, "")
