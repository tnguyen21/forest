# Trie

Implementation of a Trie that considers edit distance.

## Project Organization

```
│   README.md                          <- Top-level README that overviews project
│   run.py                             <- Entry point script that runs an example/experiment
│───datasets                           <- .txt files which contain various datasets used in experiments
│       gazetteer_entries.txt
│       misspell_words_text.txt
│       README                         <- explains what each data set is and how they are formatted
│       wikipedia_answer_key.txt
│
│───frontend                           <- contains boilerplate create-react-app for demoing forest project
│───experiments                        <- dir which holds explanations, scripts, and sample results from small experiments
│       │───word_len_experiment
│       │       driver.py              <- script to run experiment
│       │       README.md              <- README for this particular experiment
|       │       *.pkl                  <- serialized object of results
│       │
│       └───word_len_experiment
│               driver.py
│               README.md
|               *.json                 <- JSON of results
|
├───tests                              <- tests to evaluate behavior of data structure and algorithm lives here
│       context.py                     <- allows for importing other files as modules
│       test_trie.py                   <- tests behavior of the Trie and related algorithms
│
└───trie                               <- implementation of trie lives heres
        trie.py                        <- trie class which contains the data structure and its search algorithms
        trienode.py                    <- helper class to implement trie properly
```

## How To Run

**Pre-requisites**
- Access to `tux.cs.drexel.edu` or `Ubuntu 20.04.04 (Focal Fossa)` machine
- Python version `3.8.10`
- [`venv`](https://docs.python.org/3/tutorial/venv.html)

1. Run `python3 -m venv <your-env-name>` where `<your-env-name>` is the name of the virtual environment. A new directory should be created with the name `<your-env-name>`
1. Run `source <your-env-name>/bin/activate`

```sh
# before running source <your-env-name>/bin/activate
$ source <your-env-name>/bin/activate
# after
(<your-env-name>) $
```

1. Run `git clone git@github.com:tnguyen21/forest.git`
1. Run `cd forest`
1. Run `pip install -r requirements.txt`
1. Test successful set up by running `python3 run.py`

```sh
# example output after running script
(<your-env-name>) $ python3 run.py
Number of tries in forest:  4
Phonetic map:
...
# phonetic trie search output
=====
Number of Results: 63
(<your-env-name>) $
```

## Experments

Refer to README.md file contained within each sub-directory in `./experiments/`


## References

[Integrating Approximate String Matching with Phonetic String Similarity](https://link.springer.com/chapter/10.1007/978-3-319-98398-1_12)

## Acknowledgements

TODO
