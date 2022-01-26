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
from sklearn.metrics import classification_report
import pickle
import argparse
import json
import pandas as pd
import numpy as np
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
    """
    """
    # Load trie
    trie = load_trie_from_pkl(trie_pkl_path)
    phonetic_trie = PhoneticTrie()
    phonetic_trie.add_trie(trie)

    # Load data
    train_prep_df = pd.read_csv(train_data_path)
    val_prep_df = pd.read_csv(validation_data_path)

    if data_dir == None:
        # Generate trian data set
        start_time = datetime.now()
        train_df = generate_data_set(trie, train_prep_df)
        end_time = datetime.now()
        logger_object["generate_train_data_set_time"] = (
            end_time - start_time
        ).total_seconds()

        # Generate validation data set
        start_time = datetime.now()
        val_df = generate_data_set(trie, val_prep_df)
        end_time = datetime.now()
        logger_object["generate_val_data_set_time"] = (
            end_time - start_time
        ).total_seconds()

        # Save data set
        train_df.to_csv("./train_df.csv", index=False)
        val_df.to_csv("./val_df.csv", index=False)
    else:
        # Load data set
        train_df = pd.read_csv(f"{data_dir}/train_df.csv")
        val_df = pd.read_csv(f"{data_dir}/val_df.csv")

    # Split data and labels
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    y_val = val_df["label"]
    X_val = val_df.drop(columns=["label"])

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
    logger_object["metrics"] = metrics


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
            f"seriailzed_trie_size_run_{datetime.now().timestamp()}.json", "w"
        ) as f:
            json.dump(logger_object, f, indent=2)
