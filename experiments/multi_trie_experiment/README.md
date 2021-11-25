## Multi-Trie Search Experiment

### Description

This script runs a small experiment using the Princeton
Word Net database.

This script observes the performance of the multi-trie search
by separating the dataset into two separate tries, one with
words of length 7 or less, and one with words of length
greater than 5. Search shorter words with ED=1, longer
words with ED=2.

Hoping to determine if multi-trie search with this method
runs faster than single-trie search, and by how much.

### Running Locally

TODO

There's the question of adding the correct paths to the `PYTHONPATH` variable and how to do that on different environments.

As it stands, the script can only be run locally within VS Code or with some work arounds with adding the root directory to one's `PYTHONPATH`

### Results

Results are dumped into JSON files which contain the following:

```json
{
  "query_loading_time": "how long it takes to load the queries",
  "trie_loading_time": "how long it takes to load in PWN and add nodes to Tries",
  "avg_one_trie_search_time": "avg time in (s) to search queries in one trie with all words",
  "avg_multi_trie_search_time": "avg time in (s) to search queries in two tries separated by word len",
  "total_time": "total time (s) it took script to run"
}
```

### Results Log

```
multi_trie_experiment_run_1637806704.384542: First logged off experiment run. Noticed that multi-trie search was faster by about 20%.
```
