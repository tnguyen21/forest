## Serialized Trie Size and Time Reading Experiment

### Description

This script runs a small experiment using the Princeton
Word Net database.

### Running Locally

TODO

There's the question of adding the correct paths to the `PYTHONPATH` variable and how to do that on different environments.

As it stands, the script can only be run locally within VS Code or with some work arounds with adding the root directory to one's `PYTHONPATH`

### Results Log

```
Below runs ran on local laptop. Specs are as follows:
- OS Name: Microsoft Windows 10 Home
- System Type: x64-based PC
- Processor: Intel(R) Code(TM) i7-8550U CPI @ 1.8 GHz
- RAM: 12 GB

# of bytes | filename
  812599 gazetteer_entries.txt
11274297 pwn_trie_test.pkl

serialized Trie file loaded with all PWN words is about ~13x larger,
but loading in a serialized Trie is ~100x faster than loading in the dictionary
and appending each word to the Trie.
```
