"""
TODO
"""

# TODO hacky way to add to python path -- need to find better way to mangage imports
from asyncio.log import logger
from re import A
import sys
import os

from sklearn.metrics import classification_report

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # noqa: E501


from trie import PhoneticTrie
from datetime import datetime
from common import load_trie_from_pkl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
import pickle
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

logger_object = {}


def generate_data_set(ptrie: PhoneticTrie, data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        ptrie: PhoneticTrie to use for searching
        data_df: DataFrame containing the data to be searched
            which has columns ["word", "search", "edit_distance"]
    """
    train_columns = [
        "word",
        "query",
        "dmetaphone_sim",
        "dmetaphone_ed",
        "metaphone_sim",
        "metaphone_ed",
        "nysiis_sim",
        "nysiis_ed",
        "soundex_sim",
        "soundex_ed",
        "og_sim",
        "og_ed",
        "label",
    ]

    train_df = pd.DataFrame(columns=train_columns)
    for idx, word, search, ed in data_df.itertuples():
        print(f"{idx} Searching for {search}")
        results = ptrie.search(
            search,
            metaphone_output=True,
            dmetaphone_output=True,
            soundex_output=True,
            nysiis_output=True,
        )

        #! this is slow, but it works
        for result in results:
            label = 1 if result["result"] == word else 0

            append_row = [
                word,
                search,
                result["dmetaphone_jaro_winkler_similarity"],
                result["dmetaphone_edit_distance"],
                result["metaphone_jaro_winkler_similarity"],
                result["metaphone_edit_distance"],
                result["nysiis_jaro_winkler_similarity"],
                result["nysiis_edit_distance"],
                result["soundex_jaro_winkler_similarity"],
                result["soundex_edit_distance"],
                result["original_jaro_winkler_similarity"],
                result["original_edit_distance"],
                label,
            ]

            # append row to df
            train_df.loc[len(train_df)] = append_row

    return train_df


def main(
    trie_pkl_path: str, train_data_path: str, validation_data_path: str, data_dir: str
):
    """ """
    # Load phonetic_trie
    phonetic_trie = load_trie_from_pkl(trie_pkl_path)
    #! Manually setting trie edit distance here for experiments, probably don't want
    #! to do this in the future
    phonetic_trie.tries[0]["trie"].max_edit_distance = 3
    logger_object["trie_edit_distance"] = 3

    # Load data
    train_prep_df = pd.read_csv(train_data_path)
    val_prep_df = pd.read_csv(validation_data_path)

    if data_dir == None:
        # Generate trian data set
        start_time = datetime.now()
        train_df = generate_data_set(phonetic_trie, train_prep_df)
        end_time = datetime.now()
        logger_object["generate_train_data_set_time"] = (
            end_time - start_time
        ).total_seconds()

        # Generate validation data set
        start_time = datetime.now()
        val_df = generate_data_set(phonetic_trie, val_prep_df)
        end_time = datetime.now()
        logger_object["generate_val_data_set_time"] = (
            end_time - start_time
        ).total_seconds()

        # Save data set
        train_df.to_csv(
            "./experiments/single_word_evaluation_task/train_df_ed3.csv", index=False
        )
        val_df.to_csv(
            "./experiments/single_word_evaluation_task/val_df_ed3.csv", index=False
        )
    else:
        # Load data set
        train_df = pd.read_csv(f"{data_dir}/train_df.csv")
        val_df = pd.read_csv(f"{data_dir}/val_df.csv")

    # Split data and labels
    X_train = train_df.drop(columns=["word", "query", "label"])
    y_train = train_df["label"].astype(int)
    X_val = val_df.drop(columns=["word", "query", "label"])
    y_val = val_df["label"].astype(int)

    # Train model
    classifier = LogisticRegression()
    train_start_time = datetime.now()
    classifier.fit(X_train, y_train)
    train_end_time = datetime.now()
    logger_object["train_time"] = (train_end_time - train_start_time).total_seconds()

    # Serialize and save model
    with open("./model.pkl", "wb") as f:
        pickle.dump(classifier, f)

    # Calculate metrics
    y_pred = classifier.predict(X_val)
    metrics = classification_report(y_val, y_pred, output_dict=True)
    logger_object["metrics_w_phonetics"] = metrics

    # -- Compare models and performance for non-phonetic representations --
    train_no_phonetics_df = train_df.drop(
        # will only have jw sim and ed for original word and results
        columns=[
            "dmetaphone_sim",
            "dmetaphone_ed",
            "metaphone_sim",
            "metaphone_ed",
            "nysiis_sim",
            "nysiis_ed",
            "soundex_sim",
            "soundex_ed",
        ]
    )

    val_no_phonetics_df = val_df.drop(
        columns=[
            "dmetaphone_sim",
            "dmetaphone_ed",
            "metaphone_sim",
            "metaphone_ed",
            "nysiis_sim",
            "nysiis_ed",
            "soundex_sim",
            "soundex_ed",
        ]
    )

    X_no_phonetics_train = train_no_phonetics_df.drop(
        columns=["word", "query", "label"]
    )
    y_no_phonetics_train = train_no_phonetics_df["label"].astype(int)
    X_no_phonetics_val = val_no_phonetics_df.drop(columns=["word", "query", "label"])
    y_no_phonetics_val = val_no_phonetics_df["label"].astype(int)

    no_phonetics_classifier = LogisticRegression()
    no_phonetics_train_start_time = datetime.now()
    no_phonetics_classifier.fit(X_no_phonetics_train, y_no_phonetics_train)
    no_phonetics_train_end_time = datetime.now()
    logger_object["no_phonetics_train_time"] = (
        no_phonetics_train_end_time - no_phonetics_train_start_time
    ).total_seconds()

    y_no_phonetics_pred = no_phonetics_classifier.predict(X_no_phonetics_val)
    metrics_no_phonetics = classification_report(
        y_no_phonetics_val, y_no_phonetics_pred, output_dict=True
    )
    logger_object["metrics_no_phonetics"] = metrics_no_phonetics

    # -- Compare models and performance w only dmetaphone --
    train_dmetaphone_df = train_df.drop(
        columns=[
            "metaphone_sim",
            "metaphone_ed",
            "nysiis_sim",
            "nysiis_ed",
            "soundex_sim",
            "soundex_ed",
        ]
    )

    val_dmetaphone_df = val_df.drop(
        columns=[
            "metaphone_sim",
            "metaphone_ed",
            "nysiis_sim",
            "nysiis_ed",
            "soundex_sim",
            "soundex_ed",
        ]
    )

    X_dmetaphone_train = train_dmetaphone_df.drop(columns=["word", "query", "label"])
    y_dmetaphone_train = train_dmetaphone_df["label"].astype(int)
    X_dmetaphone_val = val_dmetaphone_df.drop(columns=["word", "query", "label"])
    y_dmetaphone_val = val_dmetaphone_df["label"].astype(int)

    dmetaphone_classifier = LogisticRegression()
    dmetaphone_train_start_time = datetime.now()
    dmetaphone_classifier.fit(X_dmetaphone_train, y_dmetaphone_train)
    dmetaphone_train_end_time = datetime.now()
    logger_object["dmetaphone_train_time"] = (
        dmetaphone_train_end_time - dmetaphone_train_start_time
    ).total_seconds()

    y_dmetaphone_pred = dmetaphone_classifier.predict(X_dmetaphone_val)
    metrics_dmetaphone = classification_report(
        y_dmetaphone_val, y_dmetaphone_pred, output_dict=True
    )
    logger_object["metrics_dmetaphone"] = metrics_dmetaphone


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
        "--validation_data_path",
        help="Path to .csv file containing validation data",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--save_data",
        help="Boolean flag to log out .json file containing results",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--data_from_dir",
        help="Path to dir containing generated data files if they exist",
        default=None,
        required=False,
    )

    args = parser.parse_args()

    script_start_time = datetime.now()
    main(
        args.trie_pkl_path,
        args.training_data_path,
        args.validation_data_path,
        args.data_from_dir,
    )
    script_end_time = datetime.now()

    logger_object["total_runtime"] = (
        script_end_time - script_start_time
    ).total_seconds()

    pprint(logger_object)

    if args.save_data:
        with open(
            f"single_word_evaluation_run_ed3_{datetime.now().timestamp()}.json", "w"
        ) as f:
            json.dump(logger_object, f, indent=2)
