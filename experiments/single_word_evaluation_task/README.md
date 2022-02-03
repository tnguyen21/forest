## Single Word Evaluation Task

### Description

Single word evaluation task based on dataset of Federal Drug Adminstration (FDA) terms.

Given a dictionary of various FDA terms and queries which introduce typos, mispellings, and other perturbation of the orginal word, develop a system that will be able to recognize the original word. Several methodologies will be used to complete this task, and the accuracy and performance (e.g. query time) of the methodology will be reported out below

### Directory Structure

```
├── README.md
├── common.py                           <- common functions across methodlogies (e.g. data set loading)
├── edit_distance_one.py                <- TODO
├── edit_distance_two.py                <- TODO
├── exact_match.py                      <- TODO
└── single_word_evaluation.py           <- TODO

```

### Running Locally

TODO

There's the question of adding the correct paths to the `PYTHONPATH` variable and how to do that on different environments.

As it stands, the script can only be run locally within VS Code or with some work arounds with adding the root directory to one's `PYTHONPATH`

### Discussion

#### Exact Match

As expected, queries with exact match are _extremely_ fast, as pointers become null very quickly when words are not an exact match.

The other point of interest is the 2 results that are "false positives". These are words that have been modified from another word (e.g. taking our characters, typos, adding characters), and resulted in being another word in the dictionary.

This only happened for 2 queries amongst the 17.000 queries tried, so it's a fairly rare occurrence.
