## Single Word Evaluation Task

### Description

Single word evaluation task based on dataset of Federal Drug Adminstration (FDA) terms.

Given a dictionary of various FDA terms and queries which introduce typos, mispellings, and other perturbation of the orginal word, develop a system that will be able to recognize the original word. Several methodologies will be used to complete this task, and the accuracy and performance (e.g. query time) of the methodology will be reported out below

### Directory Structure

```
├── README.md
├── common.py                           <- common functions across methodlogies (e.g. data set loading)
├── evaluate.py                         <- functions used for evaluating model
├── edit_distance_one.py                <- tests avg search time for ED=1
├── edit_distance_two.py                <- tests avg search time for ED=2
├── exact_match.py                      <- tests if searchs find false-positives for ED=0
└── single_word_evaluation.py           <- script that trains models and evaluates them

```

### Running Locally

#### Single Word Evaluation Task

1. Create a `PhoneticTrie` using the script in `common.py`. Assumes there is a text file (.txt, .csv) with dictionary entry separated by newlines

```
$ python common.py \
    --data_path="./datasets/dictionary.csv" \
    --pkl_output_path="./phonetic_trie.pkl"
```

Will output `.pkl` file at `pkl_output_path` which contains serialized `PhoneticTrie` loaded with words from `data_path`

2. Train model and add to `PhoneticTrie` and evaluate performance using `single_word_evaluation.py`. Assumes there are three \*.csvs for train, validation, and testing that have 3 colums: WORD, SEARCH, EDIT_DISTANCE.

```
$ python single_word_evaluation.py \
    --trie_pkl_path="path/to/phonetic_trie.pkl" \
    --training_data_path="path/to/train.csv" \
    --validation_data_path="path/to/val.csv" \
    --test_data_path="path/to/test.csv" \
    --save_data
```

Outputs `*.csv` datasets that's used to train `LogisticRegression` models at `/experiments/single_word_evaluation_task/datasets`, the `LogisticRegression` models themselves serialized into `.pkl` files, and a `*.json` file containing metrics from the model at root.

**`single_word_evaluation.py` Usage**

There are additional CLI args for convenience.

`--datasets_dir`: Path to dir containing generated data files (if they exist)
`--min_edit_distance`: Min edit distance to use for model
`--max_edit_distance`: Max edit distance to use for model
`--train_phonetic_model`: Flag to train phonetic model
`--train_dmetaphone_model`: Flag to train dmetaphone model
`--train_no_phonetic_model`: Flag to train no phonetic model

### Discussion

#### Exact Match

As expected, queries with exact match are _extremely_ fast, as pointers become null very quickly when words are not an exact match.

The other point of interest is the 2 results that are "false positives". These are words that have been modified from another word (e.g. taking our characters, typos, adding characters), and resulted in being another word in the dictionary.

This only happened for 2 queries amongst the 17.000 queries tried, so it's a fairly rare occurrence.
