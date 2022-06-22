""" dataset_generation.py

Code which generates evaluation dataset for multi-word matching task.

"""

from pprint import pprint
import random
import csv

from more_itertools import distinct_combinations

filler_words = ["uh", "um", "er", "ah"]

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
    
    for (cuid, entry) in dictionary:
        # randomly choose how many sentences will start with entry
        number_of_sentences_that_start_with_entry = random.randint(1, 3)
        # randomly choose [1, 5] expressions in sentence
        number_of_additional_expressions = random.randint(1, 5)
        
        # TODO save off labels for when expressions start and end
        # ? token or character position
        # ? separating cuids
        for _ in range(number_of_sentences_that_start_with_entry):
            # begin sentence with entry
            sentence = [entry]
            
            # add additional expressions to sentence
            for _ in range(number_of_additional_expressions):
                # number of filler words between expressions in sentence
                for _ in range(random.randint(0, 5)):
                    sentence += [random.choice(filler_words)]
                
                # add another expression to the sentence, 2nd element in tuple
                sentence += [random.choice(dictionary)[1]]
        print(" ".join(sentence))


if __name__ == "__main__":
    test_file = "datasets\imdb_movie_titles\-of.csv"
    generate_datasets(test_file, "")
