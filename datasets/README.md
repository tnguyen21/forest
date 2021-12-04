These .txt files contain data sets for dictionaries that were used
in testing fast phonetic search with a Trie that also considers
Edit Distance.

`gazetteer_entries.txt` contains a list of input words to place in
a Trie structure.

`gazetteer_entries_subset.txt` contains a list of words that have been selected as a subset of words to query for while performing experiments. The list is constructed of words that vary from length 4-9, eight words each.

`misspell_words_text.txt` contains the misspelled variants
of words based on the [Commonly Misspelled English words page](https://en.wikipedia.org/wiki/Commonly_misspelled_English_words).

`wikipedia_answer_key.txt` contains commonly misspelled words and the correct
spelling separated by a semi-colon.

`all_drugs.csv` is a list of ~5000 drug names used in clinical settings. These drug names are misspelled fairly often due to their length and complexity, and good candidates to test the performance and correctness of our fuzzy search implementation.
