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
Below runs ran on local laptop. Specs are as follows:
- OS Name: Microsoft Windows 10 Home
- System Type: x64-based PC
- Processor: Intel(R) Code(TM) i7-8550U CPI @ 1.8 GHz
- RAM: 12 GB

Run 1636745299.1965 is a first run without logging off search times.

Run 1637022455.73717 logs off search times pre-any optimization to the search algorithm

word_len_experiment_run_1637639029.680423 ran from PowerShell, as opposed to running within VS Code shell. Additionally, used modified `search()` method which does not append node to active nodes list if ED > MAX_ED. The `search()` method here also includes a `max_jaro_distance` metric which filters search reuslts. Drastically reduced amount of time it took search to run. Also changed the average amount of results found for 4 letter words with ED = 2.

word_len_experiment_run_1637639442.696894 ran on VS Code shell (which is just an instance of PowerShell, now that I think about it). Similar run times.
```
