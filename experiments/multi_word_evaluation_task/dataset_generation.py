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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn import preprocessing


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

    for _, id, expression in tqdm(
        dict_df.itertuples(), desc="Load dictionary into Forest..."
    ):
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


def generate_sample_sentences(dictionary_path: str, dataset_output_path: str,) -> None:
    """
    given path to dictionary with cuid and term,
    generate a dataset to use for the multi-word matching task.
    outputs dictionary as a csv
    """
    # read csv into List[Tuple[str, str]]
    with open(dictionary_path, "r") as f:
        reader = csv.reader(f)
        dictionary = [tuple(row) for row in list(reader)]

    # create set of unique words from dictionary
    dictionary_terms_set = set()
    for (_, entry) in dictionary:
        # assume words in entry are separated by space
        for word in entry.split(" "):
            dictionary_terms_set.add(word)

    # convert set->list for ease of use later
    dictionary_terms = list(dictionary_terms_set)

    sample_sentences = {}

    for (_, entry) in tqdm(dictionary, desc="Generating sample sentences..."):
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
                start_token_idx = len(sentence)
                sentence += new_expression.split(" ")
                end_token_idx = len(sentence)
                annotations.append([start_token_idx, end_token_idx, new_expression])

            # write out sentence and annotations to dataset
            joined_sentence = " ".join(sentence)
            sample_sentences[joined_sentence] = []
            for annotation in annotations:
                sample_sentences[joined_sentence].append(annotation)

    with open(dataset_output_path, "w") as f:
        json.dump(sample_sentences, f, indent=2)

def generate_dataset(
    sample_sentence_dataset_path: str, forest_pkl_path: str, search_window: int, output_dataset_path: str
):
    """
    Generate dataset to use as input for training logistic regression
    model for recognizing the beginning of multi-word expressions
    """
    # create fd for writing to csv
    
    output_csv_file = open(output_dataset_path, "w", newline="\n")
    output_csv_writer = csv.writer(output_csv_file, delimiter=",")

    # write header
    train_df_col = [
        "expression_labels",
        "sentence",
        "matched_token",
        "t1_len_expression",
        "t1_len_result_word",
        "t1_position_in_expression",
        "t1_word_determining_score",
        "t1_cuid_determining_score",
        "t2_len_expression",
        "t2_len_result_word",
        "t2_position_in_expression",
        "t2_word_determining_score",
        "t2_cuid_determining_score",
        "t3_len_expression",
        "t3_len_result_word",
        "t3_position_in_expression",
        "t3_word_determining_score",
        "t3_cuid_determining_score",
        "t4_len_expression",
        "t4_len_result_word",
        "t4_position_in_expression",
        "t4_word_determining_score",
        "t4_cuid_determining_score",
        "t5_len_expression",
        "t5_len_result_word",
        "t5_position_in_expression",
        "t5_word_determining_score",
        "t5_cuid_determining_score",
        "label",
    ]

    output_csv_writer.writerow(train_df_col)

    # loading in dataset and serialized fores
    with open(sample_sentence_dataset_path, "r") as f:
        sample_sentence_dict = json.loads(f.read())

    forest = None
    with open(forest_pkl_path, "rb") as f:
        forest = pickle.load(f)

    for sentence, expression_list in tqdm(sample_sentence_dict.items(), desc="Querying Forest and generating LR training set..."):
    # for sentence, expression_list in sample_sentence_dict.items():
        token_concept_dictionary = forest.get_token_concept_dictionary(sentence)
        token_concept_tuples = []
        # convert dict into list of tuples to preserve order and make lookups faster
        for token, matches in token_concept_dictionary.items():
            token_concept_tuples.append((token, matches))
        
        # pad beginning and end of tokens with None based on our search window
        # to be used for determining if token is in phrase or not
        # ! this is bespoke and hacky. should figure more elegant solution, or at least clean up this code
        token_concept_tuples = [(None, None) for _ in range(search_window)] + token_concept_tuples + [(None, None) for _ in range(search_window)]

        for idx in range(2, len(token_concept_tuples)-2):
            # TODO explain bespoke logic here
            search_token_window = token_concept_tuples[idx-search_window:idx+search_window+1]
            for match in token_concept_tuples[idx][1]:
                # print("sentence", sentence)
                # print("match", match)
                training_row = [expression_list, sentence, str(match)]
                m_result_word = match[0]
                m_expression = match[1]
                m_concept_id = match[2]
                for token in search_token_window:
                    matching_cuid = False
                    # check if beginning or end of sentence
                    if token[1] == None:
                        # mark with -1s
                        training_row += [-1, -1, -1, -1, -1]
                        continue

                    # check if token in window has matching concept id 
                    for _, _, concept_id, expr_len, result_len, token_position, word_det_score, cuid_det_score in token[1]:
                        if m_concept_id == concept_id:
                            matching_cuid = True
                            training_row += [expr_len, result_len, token_position, word_det_score, cuid_det_score]
                            break

                    if not matching_cuid:
                        training_row += [0, 0, 0, 0, 0]
                
                # add label
                correct = False
                split_sentence = sentence.split(" ")
                for start_idx, _, expression in expression_list:
                    if (m_expression == expression) and (split_sentence[start_idx] == m_result_word):
                        correct = True
                        break
                if correct:
                    training_row += [1]
                else:
                    training_row += [0]
                # print(training_row)
                output_csv_writer.writerow(training_row)
                training_row = []

    # don't forget to close open fd!
    output_csv_file.close()


if __name__ == "__main__":
    # write header
    train_df_col = [
        "expression_labels",
        "sentence",
        "matched_token",
        "t1_len_expression",
        "t1_len_result_word",
        "t1_position_in_expression",
        "t1_word_determining_score",
        "t1_cuid_determining_score",
        "t2_len_expression",
        "t2_len_result_word",
        "t2_position_in_expression",
        "t2_word_determining_score",
        "t2_cuid_determining_score",
        "t3_len_expression",
        "t3_len_result_word",
        "t3_position_in_expression",
        "t3_word_determining_score",
        "t3_cuid_determining_score",
        "t4_len_expression",
        "t4_len_result_word",
        "t4_position_in_expression",
        "t4_word_determining_score",
        "t4_cuid_determining_score",
        "t5_len_expression",
        "t5_len_result_word",
        "t5_position_in_expression",
        "t5_word_determining_score",
        "t5_cuid_determining_score",
        "label",
    ]
    # dictionary_input_path = "datasets/nasa_shared_task/HEXTRATO_dictionary.csv"
    train_dictionary_input_path = "datasets/umls_small_dictionary/training.csv"
    tuning_dictionary_input_path = "datasets/umls_small_dictionary/tuning.csv"
    test_dictionary_input_path = "datasets/umls_small_dictionary/test.csv"
    # train_dictionary_input_path = "datasets/nasa_shared_task/training.csv"
    # tuning_dictionary_input_path = "datasets/nasa_shared_task/tuning.csv"
    # test_dictionary_input_path = "datasets/nasa_shared_task/test.csv"
    # dictionary_input_path = "datasets/imdb_movie_titles/-a.csv"
    sample_sentence_output_path = "experiments/multi_word_evaluation_task/"
    lr_model_dataset_path = "experiments/multi_word_evaluation_task/train_data.csv"
    tuning_lr_model_dataset_path = "experiments/multi_word_evaluation_task/tuning_data.csv"
    test_lr_model_dataset_path = "experiments/multi_word_evaluation_task/test_data.csv"
    forest_output_path = "test_forest.pkl"

    create_forest(train_dictionary_input_path, forest_output_path)
    generate_sample_sentences(train_dictionary_input_path, sample_sentence_output_path + "train_sample_sentences.json")
    generate_sample_sentences(tuning_dictionary_input_path, sample_sentence_output_path + "tuning_sample_sentences.json")
    generate_sample_sentences(test_dictionary_input_path, sample_sentence_output_path + "test_sample_sentences.json")
    generate_dataset(sample_sentence_output_path + "train_sample_sentences.json", forest_output_path, 2, lr_model_dataset_path)
    generate_dataset(sample_sentence_output_path + "tuning_sample_sentences.json", forest_output_path, 2, tuning_lr_model_dataset_path)
    generate_dataset(sample_sentence_output_path + "test_sample_sentences.json", forest_output_path, 2, test_lr_model_dataset_path)

    train_df = pd.read_csv(lr_model_dataset_path, delimiter=",", names=train_df_col, header=0)
    # print(train_df.head())
    X_train = train_df.drop(columns=["expression_labels", "sentence", "matched_token", "label"])
    # print(X_train.head())
    y_train = train_df["label"].astype(int)
    # print(y_train.head())

    # had to adjust max_iter
    # getting warning about failing to converge at default for 100 iters and 1000 iters.
    # warning goes away at 5000 iters
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    print("train score", classifier.score(X_train, y_train))
    tuning_df = pd.read_csv(tuning_lr_model_dataset_path, delimiter=",", names=train_df_col, header=0)
    X_tuning = tuning_df.drop(columns=["expression_labels", "sentence", "matched_token", "label"])
    y_tuning = tuning_df["label"].astype(int)
    
    test_df = pd.read_csv(test_lr_model_dataset_path, delimiter=",", names=train_df_col, header=0)
    X_test = test_df.drop(columns=["expression_labels", "sentence", "matched_token", "label"])
    y_test = test_df["label"].astype(int)

    y_pred = classifier.predict_proba(X_test)[:, 1]
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    model_f1_score = f1_score(y_test, y_pred)
    print(f"Model f1 score at threshold 0.5: {model_f1_score}")

    y_pred = classifier.predict_proba(X_test)[:, 1]
    y_pred[y_pred >= 0.75] = 1
    y_pred[y_pred < 0.75] = 0
    model_f1_score = f1_score(y_test, y_pred)
    print(f"Model f1 score at threshold 0.75: {model_f1_score}")

    y_pred = classifier.predict_proba(X_test)[:, 1]
    y_pred[y_pred >= 0.9] = 1
    y_pred[y_pred < 0.9] = 0
    model_f1_score = f1_score(y_test, y_pred)
    print(f"Model f1 score at threshold 0.9: {model_f1_score}")
