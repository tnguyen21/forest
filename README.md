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

TODO

TODO - note about pre-commit

## Experments

TODO

## Results

TODO

## Future Work

TODO

## References

TODO

## Acknowledgements

TODO
