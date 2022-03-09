""" evaluate.py

TODO

"""

from typing import Union
from trie import PhoneticTrie
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd


def compute_metrics(
    data_df: pd.DataFrame,
    search_edit_distance: int,
    phonetic_trie: PhoneticTrie,
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

    # set up ptrie
    phonetic_trie.set_logistic_regression_model(classifier)
    phonetic_trie.logistic_regression_model_threshold = threshold
    if threshold:
        use_model = True
    else:
        use_model = False

    # query trie and compute scores
    for _, target_word, search, _ in tqdm(
        data_df.itertuples(), ascii=True, desc="Computing metrics"
    ):
        results = phonetic_trie.search(
            search, max_edit_distance=search_edit_distance, use_lr_model=use_model
        )

        result_words_list = [result["result"] for result in results]
        if target_word not in result_words_list:
            fn += 1
        for idx, result in enumerate(results):
            rank = idx + 1
            if result["result"] == target_word:
                tp += 1 / rank
            elif result != target_word:
                fp += 1 / rank

    print("tp", tp)
    print("fp", fp)
    print("fn", fn)

    metrics = {}
    try:
        metrics["precision"] = tp / (tp + fp)
        metrics["recall"] = tp / (tp + fn)
        metrics["f1"] = (
            2
            * metrics["precision"]
            * metrics["recall"]
            / (metrics["precision"] + metrics["recall"])
        )
    except ZeroDivisionError:
        print("Divide by zero error")
        metrics["tp"] = tp
        metrics["fp"] = fp
        metrics["fn"] = fn
    return metrics
