## Word Length and Query Results Experiment

### Description

This script runs a small experiment using the Princeton
Word Net database.

Words of varying length are loaded into a Trie, and then we
query words of varying lengths and various edit distances.

Results should approximately inform us of how many results
may turn up as a result of word length and edit distance. These
results can be a proxy in telling us how difficult it may be
to achieve performance in the NER task.

### Running Locally

TODO

### Deserializing Results

To de-serialize a `.pkl` file and then view results from a particular run, reference the following code snippet.

```python
import pickle

with open("results.pkl", "rb") as f: # note we open the file in binary
    results = pickle.load(f)

print(results.keys()) # view what data has been logged off
print(results["<key>"])
```
