## Multi Word Evaluation Task

### Description

`dataset_generation.py` is a script which is used to construct a Forest, generate a dataset to train a Logistic Regression model, then train that Logistic Regression model, and log all artifcats out.
Paths to these artifacts can be modified within the script.

`multi_word_evaluation.py` is a script which loads the Forest and Logistic Regression model, and then performs some queries with strings that can be modified within the script.
Output is printed out to stdin.

### Running Locally

**Make sure you have activated your virtual environment**

`$ source /<env-name>/bin/activate`

**Run this from the root of the project directory**

```
# generate datasets, forest, and logistic regression models
$ python3 experiments/multi_word_evaluation_task/dataset_generation.py
# some output will be printed out to stdin
$ python3 experiments/multi_word_evaluation_task/multi_word_evaluation.py
```

### Results Log
