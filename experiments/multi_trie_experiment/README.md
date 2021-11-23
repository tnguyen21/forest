## Multi-Trie Search Experiment

### Description

This script runs a small experiment using the Princeton
Word Net database.

TODO

### Running Locally

TODO

There's the question of adding the correct paths to the `PYTHONPATH` variable and how to do that on different environments.

As it stands, the script can only be run locally within VS Code or with some work arounds with adding the root directory to one's `PYTHONPATH`

### Deserializing Results

To de-serialize a `.pkl` file and then view results from a particular run, reference the following code snippet.

```python
import pickle

with open("results.pkl", "rb") as f: # note we open the file in binary
    results = pickle.load(f)

print(results.keys()) # view what data has been logged off
print(results["<key>"])
```

### Results Log

```

```
