"""
TODO
"""

# TODO hacky way to add to python path -- need to find better way to mangage imports
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # noqa: E501


from datetime import datetime
from common import load_trie_from_pkl, generate_data_set
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score
from tqdm import tqdm
import dill as pickle
from phonetics import dmetaphone  # needed to load in trie
import argparse
import json
import numpy as np
import pandas as pd
from pprint import pprint
from evaluate import compute_metrics


logger_object = {}


def train_phonetic_model(
    trie_pkl_path: str,
    train_data_path: str,
    validation_data_path: str,
    test_data_path: str,
    dataset_dir: str,
    edit_distance: int,  # this ED is temp
):
    # Load phonetic_trie
    phonetic_trie = load_trie_from_pkl(trie_pkl_path)
    logger_object["trie_edit_distance"] = edit_distance

    #! when word is searched and no result if found, this should be a penalty against scoring
    #! it's false

    # Load data
    print("Reading in data...")
    train_prep_df = pd.read_csv(train_data_path)
    val_prep_df = pd.read_csv(validation_data_path)
    test_prep_df = pd.read_csv(test_data_path)

    # If datasets already generated, then load them in
    if dataset_dir:
        train_df = pd.read_csv(
            f"./experiments/single_word_evaluation_task/datasets/train_df_ed{edit_distance}_phonetic.csv"
        )
        val_df = pd.read_csv(
            f"./experiments/single_word_evaluation_task/datasets/val_df_ed{edit_distance}_phonetic.csv"
        )
    # Generate data sets and save them
    else:
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

    # only use prediction probability for positive label
    y_pred = classifier.predict_proba(X_val)[:, 1]
    # precision, recall, thresholds are np.arrays
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
    f1_scores = (2 * precision * recall) / (precision + recall)

    idx = np.argmax(f1_scores)
    best_threshold = thresholds[idx]

    print(f"Best threshold: {best_threshold}, best f1 score: {f1_scores[idx]}")

    # Calculate model f1 score
    y_pred = classifier.predict_proba(X_val)[:, 1]
    y_pred[y_pred >= best_threshold] = 1
    y_pred[y_pred < best_threshold] = 0
    model_f1_score = f1_score(y_val, y_pred)
    print(f"Model f1 score: {model_f1_score}")
    logger_object["phonetic_model_f1_score"] = model_f1_score

    # Calculate metrics
    print("Computing performance metrics on test set...")
    # ! this classifier is trained on _training_ data
    metrics_no_tuning = compute_metrics(
        test_prep_df, edit_distance, phonetic_trie=phonetic_trie, classifier=classifier
    )
    metrics_w_tuning = compute_metrics(
        test_prep_df,
        edit_distance,
        classifier=classifier,
        phonetic_trie=phonetic_trie,
        threshold=best_threshold,
    )
    metrics_w_tuning["best_threshold"] = best_threshold

    logger_object["metrics_w_phonetic_and_tuning"] = metrics_w_tuning
    logger_object["metrics_w_phonetic_no_tuning"] = metrics_no_tuning


def train_no_phonetic_model(
    trie_pkl_path: str,
    train_data_path: str,
    validation_data_path: str,
    test_data_path: str,
    dataset_dir: str,
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
    print("Reading in data...")
    train_prep_df = pd.read_csv(train_data_path)
    val_prep_df = pd.read_csv(validation_data_path)
    test_prep_df = pd.read_csv(test_data_path)

    # If datasets already generated, then load them in
    if dataset_dir:
        train_df = pd.read_csv(
            f"./experiments/single_word_evaluation_task/datasets/train_df_ed{edit_distance}_phonetic.csv"
        )
        val_df = pd.read_csv(
            f"./experiments/single_word_evaluation_task/datasets/val_df_ed{edit_distance}_phonetic.csv"
        )
    # Generate train data set
    else:
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

    # only use prediction probability for positive label
    y_pred = classifier.predict_proba(X_val)[:, 1]
    # precision, recall, thresholds are np.arrays
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
    f1_scores = (2 * precision * recall) / (precision + recall)

    idx = np.argmax(f1_scores)
    best_threshold = thresholds[idx]

    # Calculate model f1 score
    y_pred = classifier.predict_proba(X_val)[:, 1]
    y_pred[y_pred >= best_threshold] = 1
    y_pred[y_pred < best_threshold] = 0
    model_f1_score = f1_score(y_val, y_pred)
    print(f"Model f1 score: {model_f1_score}")
    logger_object["no_phonetic_model_f1_score"] = model_f1_score

    # Calculate metrics
    print("Computing performance metrics on test set...")
    # ! this classifier is trained on _training_ data
    metrics_no_tuning = compute_metrics(
        test_prep_df, edit_distance, phonetic_trie=phonetic_trie, classifier=classifier
    )
    metrics_w_tuning = compute_metrics(
        test_prep_df,
        edit_distance,
        phonetic_trie=phonetic_trie,
        classifier=classifier,
        threshold=best_threshold,
    )
    metrics_w_tuning["best_threshold"] = best_threshold

    logger_object["metrics_w_no_phonetic_and_tuning"] = metrics_w_tuning
    logger_object["metrics_w_no_phonetic_no_tuning"] = metrics_no_tuning


def train_dmetaphone_model(
    trie_pkl_path: str,
    train_data_path: str,
    validation_data_path: str,
    test_data_path: str,
    dataset_dir: str,
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
    test_prep_df = pd.read_csv(test_data_path)

    # If datasets already generated, then load them in
    if dataset_dir:
        train_df = pd.read_csv(
            f"./experiments/single_word_evaluation_task/datasets/train_df_ed{edit_distance}_phonetic.csv"
        )
        val_df = pd.read_csv(
            f"./experiments/single_word_evaluation_task/datasets/val_df_ed{edit_distance}_phonetic.csv"
        )
    # Generate train data set
    else:
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

    # only use prediction probability for positive label
    y_pred = classifier.predict_proba(X_val)[:, 1]
    # precision, recall, thresholds are np.arrays
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
    f1_scores = (2 * precision * recall) / (precision + recall)

    idx = np.argmax(f1_scores)
    best_threshold = thresholds[idx]

    print(f"Best threshold: {best_threshold}, best f1 score: {f1_scores[idx]}")

    # Calculate model f1 score
    y_pred = classifier.predict_proba(X_val)[:, 1]
    y_pred[y_pred >= best_threshold] = 1
    y_pred[y_pred < best_threshold] = 0
    model_f1_score = f1_score(y_val, y_pred)
    print(f"Model f1 score: {model_f1_score}")
    logger_object["dmetaphone_only_model_f1_score"] = model_f1_score

    # Calculate metrics
    print("Computing performance metrics on test set...")
    # ! this classifier is trained on _training_ data
    metrics_no_tuning = compute_metrics(
        test_prep_df, edit_distance, phonetic_trie=phonetic_trie, classifier=classifier
    )
    metrics_w_tuning = compute_metrics(
        test_prep_df,
        edit_distance,
        phonetic_trie=phonetic_trie,
        classifier=classifier,
        threshold=best_threshold,
    )
    metrics_w_tuning["best_threshold"] = best_threshold

    logger_object["metrics_dmetaphone_and_tuning"] = metrics_w_tuning
    logger_object["metrics_dmetaphone_no_tuning"] = metrics_no_tuning


def main(
    trie_pkl_path: str,
    train_data_path: str,
    validation_data_path: str,
    test_data_path: str,
    datasets_dir: str,
    train_phonetic_model_flag: bool,
    train_dmetaphone_model_flag: bool,
    train_no_phonetic_model_flag: bool,
    edit_distance: int,
):
    """ """
    # train phonetic model
    if train_phonetic_model_flag:
        train_phonetic_model(
            trie_pkl_path=trie_pkl_path,
            train_data_path=train_data_path,
            validation_data_path=validation_data_path,
            test_data_path=test_data_path,
            dataset_dir=datasets_dir,
            edit_distance=edit_distance,
        )

    # train dmetaphone model
    if train_dmetaphone_model_flag:
        train_dmetaphone_model(
            trie_pkl_path=trie_pkl_path,
            train_data_path=train_data_path,
            validation_data_path=validation_data_path,
            test_data_path=test_data_path,
            dataset_dir=datasets_dir,
            edit_distance=edit_distance,
        )

    # train no phonetic model
    if train_no_phonetic_model_flag:
        train_no_phonetic_model(
            trie_pkl_path=trie_pkl_path,
            train_data_path=train_data_path,
            validation_data_path=validation_data_path,
            test_data_path=test_data_path,
            dataset_dir=datasets_dir,
            edit_distance=edit_distance,
        )


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
        "--test_data_path",
        help="Path to .csv file containing test data",
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
        "--datasets_dir",
        help="Path to dir containing generated data files (if they exist)",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--min_edit_distance",
        help="Min edit distance to use for model",
        default=0,
        required=False,
    )
    parser.add_argument(
        "--max_edit_distance",
        help="Max edit distance to use for model",
        default=2,
        required=False,
    )
    parser.add_argument(
        "--train_phonetic_model",
        help="Flag to train phonetic model",
        default=True,
        required=False,
    )
    parser.add_argument(
        "--train_dmetaphone_model",
        help="Flag to train dmetaphone model",
        default=True,
        required=False,
    )
    parser.add_argument(
        "--train_no_phonetic_model",
        help="Flag to train no phonetic model",
        default=True,
        required=False,
    )

    args = parser.parse_args()

    # ignore noisy warnings from sklearn
    # ! not safe (probably)
    def warn(*args, **kwargs):
        pass

    import warnings

    warnings.warn = warn

    for ed in range(args.min_edit_distance, args.max_edit_distance + 1):
        script_start_time = datetime.now()
        main(
            args.trie_pkl_path,
            args.training_data_path,
            args.validation_data_path,
            args.test_data_path,
            args.datasets_dir,
            args.train_phonetic_model,
            args.train_dmetaphone_model,
            args.train_no_phonetic_model,
            ed,
        )
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
