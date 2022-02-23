"""
TODO
"""

# TODO hacky way to add to python path -- need to find better way to mangage imports
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # noqa: E501


from typing import List, Union
from trie import PhoneticTrie
from datetime import datetime
from common import load_trie_from_pkl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dill as pickle
from phonetics import dmetaphone  # needed to load in trie
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

logger_object = {}


def generate_data_set(
    ptrie: PhoneticTrie, data_df: pd.DataFrame, edit_distance: int = 2
) -> pd.DataFrame:
    """
    Args:
        ptrie: PhoneticTrie to use for searching
        data_df: DataFrame containing the data to be searched
            which has columns ["word", "search", "edit_distance"]
    """
    train_columns = [
        "target_word",
        "result_word",
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

    to_df_list = []
    for idx, word, search, ed in data_df.itertuples():
        print(f"{idx} Searching for {search} with edit distance {edit_distance}")
        results = ptrie.search(
            search,
            max_edit_distance=edit_distance,
            metaphone_output=True,
            dmetaphone_output=True,
            soundex_output=True,
            nysiis_output=True,
        )
        for result in results:
            #! if result is empty, should still add a row when creating a dataset
            #! failed searched
            label = 1 if result["result"] == word else 0

            append_row = [
                word,
                result["result"],
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
            to_df_list.append(append_row)

    train_df = pd.DataFrame(to_df_list, columns=train_columns)

    return train_df


def compute_metrics(
    data_df: pd.DataFrame,
    infer_columns: List[str],
    classifier: LogisticRegression,
    threshold: Union[float, None] = None,
) -> dict:
    """
    Compute the metrics for the given true and predicted labels
    Args:
        data_df: DataFrame containing the true and predicted labels
        infer_columns: list of columns to use for computing metrics
        classifier: sklearn classifier to use for computing metrics
    Returns:
        accuracy, precision, recall, f1
    """
    # Calculate metrics
    tp, fp, fn = 0, 0, 0
    queries = set(data_df["query"].tolist())

    for query in queries:
        results = pd.DataFrame(data_df[data_df["query"] == query])
        # use [:, 1] to only get probabilities for positive label
        results["predict_proba"] = classifier.predict_proba(results[infer_columns])[
            :, 1
        ]

        # thresholding
        # TODO move this to be a part of the phonetic trie if a LR model is given
        if threshold is not None:
            results = results[results["predict_proba"] >= threshold]

        # rank results based on probability scores
        results["rank"] = results["predict_proba"].rank(ascending=False)

        for idx, row in results.iterrows():
            if row["result_word"] == "":
                # false negative
                # FN occurs when the result is empty
                fn += 1
            elif row["target_word"] == row["result_word"]:
                # true positive
                tp += 1 / row["rank"]
            elif row["target_word"] != row["result_word"]:
                # false positive
                fp += 1 / row["rank"]

    metrics = {}
    metrics["precision"] = tp / (tp + fp)
    metrics["recall"] = tp / (tp + fn)
    metrics["f1"] = (
        2
        * metrics["precision"]
        * metrics["recall"]
        / (metrics["precision"] + metrics["recall"])
    )

    return metrics


def train_phonetic_model(
    trie_pkl_path: str,
    train_data_path: str,
    validation_data_path: str,
    edit_distance: int,  # this ED is temp
):
    # Load phonetic_trie
    phonetic_trie = load_trie_from_pkl(trie_pkl_path)
    logger_object["trie_edit_distance"] = edit_distance

    #! when word is searched and no result if found, this should be a penalty against scoring
    #! it's false

    # Load data
    train_prep_df = pd.read_csv(train_data_path)
    val_prep_df = pd.read_csv(validation_data_path)

    # Generate trian data set
    start_time = datetime.now()
    train_df = generate_data_set(phonetic_trie, train_prep_df, edit_distance)
    end_time = datetime.now()
    logger_object["generate_phonetic_train_data_set_time"] = (
        end_time - start_time
    ).total_seconds()

    # Generate validation data set
    start_time = datetime.now()
    val_df = generate_data_set(phonetic_trie, val_prep_df, edit_distance)
    end_time = datetime.now()
    logger_object["generate_phonetic_val_data_set_time"] = (
        end_time - start_time
    ).total_seconds()

    # Save phonetic data set
    train_df.to_csv(
        f"./experiments/single_word_evaluation_task/datasets/train_df_ed{edit_distance}_phonetic.csv",
        index=False,
    )
    val_df.to_csv(
        f"./experiments/single_word_evaluation_task/datasets/val_df_ed{edit_distance}_phonetic.csv",
        index=False,
    )

    # Split data and labels
    X_train = train_df.drop(columns=["target_word", "result_word", "query", "label"])
    y_train = train_df["label"].astype(int)
    X_val = val_df.drop(columns=["target_word", "result_word", "query", "label"])
    y_val = val_df["label"].astype(int)

    # Train model
    classifier = LogisticRegression()
    train_start_time = datetime.now()
    classifier.fit(X_train, y_train)
    train_end_time = datetime.now()
    logger_object["train_time"] = (train_end_time - train_start_time).total_seconds()

    # Serialize and save model
    with open(
        f"./experiments/single_word_evaluation_task/models/phonetic_model_ed_{edit_distance}.pkl",
        "wb",
    ) as f:
        pickle.dump(classifier, f)

    # find optimal threshold for LR model using val dataset
    print("Finding best threshold for LR model...")
    thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = []
    for threshold in thresholds:
        metrics = compute_metrics(val_df, X_val.columns, classifier, threshold)
        f1_scores.append(metrics["f1"])

    idx = np.argmax(f1_scores)
    best_threshold = thresholds[idx]

    print(f"Best threshold: {best_threshold}, best f1 score: {f1_scores[idx]}")

    # Calculate metrics
    print("Computing performance metrics...")
    metrics_no_tuning = compute_metrics(val_df, X_val.columns, classifier)
    metrics_w_tuning = compute_metrics(
        val_df, X_val.columns, classifier, best_threshold
    )
    metrics_w_tuning["best_threshold"] = best_threshold

    logger_object["metrics_w_phonetic_and_tuning"] = metrics_w_tuning
    logger_object["metrics_w_phonetic_no_tuning"] = metrics_no_tuning


def train_no_phonetic_model(
    trie_pkl_path: str,
    train_data_path: str,
    validation_data_path: str,
    edit_distance: int,  # this ED is temp
):
    # Load phonetic_trie
    phonetic_trie = load_trie_from_pkl(trie_pkl_path)

    # Drop phonetic tries
    # tries are held in list
    # [no_phonetic_trie, dmetaphone_trie, metaphone_trie, nysiis_trie, soundex_trie]
    # only take the first one
    phonetic_trie.tries = phonetic_trie.tries[0:1]

    # Log ED we're searching for this run
    logger_object["trie_edit_distance"] = edit_distance

    # Load data
    train_prep_df = pd.read_csv(train_data_path)
    val_prep_df = pd.read_csv(validation_data_path)

    # Generate trian data set
    start_time = datetime.now()
    train_df = generate_data_set(phonetic_trie, train_prep_df, edit_distance)
    end_time = datetime.now()
    logger_object["generate_no_phonetic_train_data_set_time"] = (
        end_time - start_time
    ).total_seconds()

    # Generate validation data set
    start_time = datetime.now()
    val_df = generate_data_set(phonetic_trie, val_prep_df, edit_distance)
    end_time = datetime.now()
    logger_object["generate_no_phonetic_val_data_set_time"] = (
        end_time - start_time
    ).total_seconds()

    # Drop columns that contain phonetic information
    train_df = train_df.drop(
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
    val_df = val_df.drop(
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

    # Save phonetic data set
    train_df.to_csv(
        f"./experiments/single_word_evaluation_task/datasets/train_df_ed{edit_distance}_no_phonetic.csv",
        index=False,
    )
    val_df.to_csv(
        f"./experiments/single_word_evaluation_task/datasets/val_df_ed{edit_distance}_no_phonetic.csv",
        index=False,
    )

    # Split data and labels
    X_train = train_df.drop(columns=["target_word", "result_word", "query", "label"])
    y_train = train_df["label"].astype(int)
    X_val = val_df.drop(columns=["target_word", "result_word", "query", "label"])
    y_val = val_df["label"].astype(int)

    # Train model
    classifier = LogisticRegression()
    train_start_time = datetime.now()
    classifier.fit(X_train, y_train)
    train_end_time = datetime.now()
    logger_object["train_time"] = (train_end_time - train_start_time).total_seconds()

    # Serialize and save model
    with open(
        f"./experiments/single_word_evaluation_task/models/no_phonetic_model_ed_{edit_distance}.pkl",
        "wb",
    ) as f:
        pickle.dump(classifier, f)

    # find optimal threshold for LR model using val dataset
    print("Finding best threshold for LR model...")
    thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = []
    for threshold in thresholds:
        metrics = compute_metrics(val_df, X_val.columns, classifier, threshold)
        f1_scores.append(metrics["f1"])

    idx = np.argmax(f1_scores)
    best_threshold = thresholds[idx]

    print(f"Best threshold: {best_threshold}, best f1 score: {f1_scores[idx]}")

    # Calculate metrics
    print("Computing performance metrics...")
    metrics_no_tuning = compute_metrics(val_df, X_val.columns, classifier)
    metrics_w_tuning = compute_metrics(
        val_df, X_val.columns, classifier, best_threshold
    )
    metrics_w_tuning["best_threshold"] = best_threshold

    logger_object["metrics_w_phonetic_and_tuning"] = metrics_w_tuning
    logger_object["metrics_w_phonetic_no_tuning"] = metrics_no_tuning


def train_dmetaphone_model(
    trie_pkl_path: str,
    train_data_path: str,
    validation_data_path: str,
    edit_distance: int,  # this ED is temp
):
    # Load phonetic_trie
    phonetic_trie = load_trie_from_pkl(trie_pkl_path)

    # Drop phonetic tries
    # tries are held in list
    # [no_phonetic_trie, dmetaphone_trie, metaphone_trie, nysiis_trie, soundex_trie]
    # only take the first one
    phonetic_trie.tries = phonetic_trie.tries[0:2]

    # Log ED we're searching for this run
    logger_object["trie_edit_distance"] = edit_distance

    # Load data
    train_prep_df = pd.read_csv(train_data_path)
    val_prep_df = pd.read_csv(validation_data_path)

    # Generate trian data set
    start_time = datetime.now()
    train_df = generate_data_set(phonetic_trie, train_prep_df, edit_distance)
    end_time = datetime.now()
    logger_object["generate_dmetaphone_train_data_set_time"] = (
        end_time - start_time
    ).total_seconds()

    # Generate validation data set
    start_time = datetime.now()
    val_df = generate_data_set(phonetic_trie, val_prep_df, edit_distance)
    end_time = datetime.now()
    logger_object["generate_dmetaphone_val_data_set_time"] = (
        end_time - start_time
    ).total_seconds()

    # Drop columns that contain phonetic information, except dmetaphone
    train_df = train_df.drop(
        columns=[
            "metaphone_sim",
            "metaphone_ed",
            "nysiis_sim",
            "nysiis_ed",
            "soundex_sim",
            "soundex_ed",
        ]
    )
    val_df = val_df.drop(
        columns=[
            "metaphone_sim",
            "metaphone_ed",
            "nysiis_sim",
            "nysiis_ed",
            "soundex_sim",
            "soundex_ed",
        ]
    )

    # Save phonetic data set
    train_df.to_csv(
        f"./experiments/single_word_evaluation_task/datasets/train_df_ed{edit_distance}_dmetaphone.csv",
        index=False,
    )
    val_df.to_csv(
        f"./experiments/single_word_evaluation_task/datasets/val_df_ed{edit_distance}_dmetaphone.csv",
        index=False,
    )

    # Split data and labels
    X_train = train_df.drop(columns=["target_word", "result_word", "query", "label"])
    y_train = train_df["label"].astype(int)
    X_val = val_df.drop(columns=["target_word", "result_word", "query", "label"])
    y_val = val_df["label"].astype(int)

    # Train model
    classifier = LogisticRegression()
    train_start_time = datetime.now()
    classifier.fit(X_train, y_train)
    train_end_time = datetime.now()
    logger_object["train_time"] = (train_end_time - train_start_time).total_seconds()

    # Serialize and save model
    with open(
        f"./experiments/single_word_evaluation_task/models/dmetaphone_model_ed_{edit_distance}.pkl",
        "wb",
    ) as f:
        pickle.dump(classifier, f)

    # find optimal threshold for LR model using val dataset
    print("Finding best threshold for LR model...")
    thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = []
    for threshold in thresholds:
        metrics = compute_metrics(val_df, X_val.columns, classifier, threshold)
        f1_scores.append(metrics["f1"])

    idx = np.argmax(f1_scores)
    best_threshold = thresholds[idx]

    print(f"Best threshold: {best_threshold}, best f1 score: {f1_scores[idx]}")

    # Calculate metrics
    print("Computing performance metrics...")
    metrics_no_tuning = compute_metrics(val_df, X_val.columns, classifier)
    metrics_w_tuning = compute_metrics(
        val_df, X_val.columns, classifier, best_threshold
    )
    metrics_w_tuning["best_threshold"] = best_threshold

    logger_object["metrics_w_phonetic_and_tuning"] = metrics_w_tuning
    logger_object["metrics_w_phonetic_no_tuning"] = metrics_no_tuning


def main(
    trie_pkl_path: str,
    train_data_path: str,
    validation_data_path: str,
    edit_distance: int,  # this ED is temp
):
    """ """
    # train phonetic model
    train_phonetic_model(
        trie_pkl_path=trie_pkl_path,
        train_data_path=train_data_path,
        validation_data_path=validation_data_path,
        edit_distance=edit_distance,
    )

    # # train dmetaphone model
    # train_dmetaphone_model(
    #     trie_pkl_path=trie_pkl_path,
    #     train_data_path=train_data_path,
    #     validation_data_path=validation_data_path,
    #     edit_distance=edit_distance,
    # )

    # # train no phonetic model
    # train_no_phonetic_model(
    #     trie_pkl_path=trie_pkl_path,
    #     train_data_path=train_data_path,
    #     validation_data_path=validation_data_path,
    #     edit_distance=edit_distance,
    # )


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

    for ed in range(0, 1):
        script_start_time = datetime.now()
        main(args.trie_pkl_path, args.training_data_path, args.validation_data_path, ed)
        script_end_time = datetime.now()

        logger_object["total_runtime"] = (
            script_end_time - script_start_time
        ).total_seconds()

        pprint(logger_object)

        if args.save_data:
            with open(
                f"single_word_evaluation_run_ed{ed}_{datetime.now().timestamp()}.json",
                "w",
            ) as f:
                json.dump(logger_object, f, indent=2)
