""" dataset_generation.py

Code which generates evaluation dataset for multi-word matching task.

"""

from pprint import pprint
from typing import Set, List
import random
import csv

def add_stop_words(dictionary_words: Set[str], sentence: List[str]) -> str:
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
    
    # list -> set conversion adds additional randomness since
    # sets do not have "ordering" of entries -- do we want this?
    filler_words = stop_words + pwn_stop_words + list(dictionary_words)

    # add [0, 5] filler words between expressions in sentence
    for _ in range(random.randint(0, 5)):
        sentence += [random.choice(filler_words)]
    
    return sentence

def generate_datasets(
    dictionary_path: str,
    dataset_output_path: str
) -> None:
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
    
    # create set of unique words from dictionary
    dictionary_terms_set = set()
    for (_, entry) in dictionary:
        # assume words in entry are separated by space
        for word in entry.split(' '):
            dictionary_terms_set.add(word)

    for (cuid, entry) in dictionary:
        # randomly choose how many sentences will start with entry
        number_of_sentences_that_start_with_entry = random.randint(1, 3)
        # randomly choose [1, 5] expressions in sentence
        number_of_additional_expressions = random.randint(1, 5)
        
        # save each label for annoations as a tuple of (int, int, str)
        # where it is (start token idx, end token idx, expression)
        annotations = []
        # TODO save off labels for when expressions start and end
        # ? token or character position -- token position
        # ? separating cuids -- don't mix expressions with same CUID
        for _ in range(number_of_sentences_that_start_with_entry):
            # begin sentence with entry
            sentence = entry.split(" ")
            annotations.append(
                (0, len(sentence) - 1, entry)
            )
            
            # add additional expressions to sentence
            for _ in range(number_of_additional_expressions):
                add_stop_words(dictionary_terms_set, sentence=sentence)

                # add another expression to the sentence, 2nd element in tuple
                new_expression = random.choice(dictionary)[1] 
                start_token_idx = len(sentence) - 1
                sentence += new_expression.split(" ")
                end_token_idx = len(sentence) - 1
                annotations.append(
                    (start_token_idx, end_token_idx, new_expression)
                )
    
        # print(" ".join(sentence))
        print(sentence)
        pprint(annotations)
if __name__ == "__main__":
    test_file = "datasets\imdb_movie_titles\-of.csv"
    generate_datasets(test_file, "")