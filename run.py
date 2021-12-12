""" run.py

File to act as the "driver" for the program.
For now, use this file for quick iteration and running
tiny experiments to get minimum functionality working.
"""

from trie import Trie, Forest
from pprint import pprint
from phonetics import metaphone


def mock_trie() -> Trie:
    t = Trie()
    t.add_entry("special")
    t.add_entry("specific")
    t.add_entry("spice")
    t.add_entry("expose")
    t.add_entry("external")
    t.add_entry("extreme")
    t.add_entry("airport")

    return t


def mock_forest() -> Forest:
    f = Forest()

    f.add_trie(min_entry_len=0, max_entry_len=4, max_edit_distance=1)
    f.add_trie(min_entry_len=5, max_entry_len=99, max_edit_distance=2)

    f.add_trie(
        min_entry_len=0,
        max_entry_len=4,
        max_edit_distance=1,
        phonetic_representation=metaphone,
    )
    f.add_trie(
        min_entry_len=5,
        max_entry_len=99,
        max_edit_distance=2,
        phonetic_representation=metaphone,
    )

    return f


if __name__ == "__main__":
    forest = mock_forest()
    # add first 500 words of test dictionary
    i = 0
    with open("datasets/gazetteer_entries.txt", "r") as f:
        for line in f:
            word = line.strip()
            forest.add_entry(word)
            i += 1
            if i > 100:
                break
    print("Number of tries in forest: ", len(forest.tries))
    print("Phonetic map:")
    pprint(forest.phonetic_map)
    print("=" * 5)

    query = "abandon"
    print("query: ", query)
    # TODO this phonetics library is pretty buggy, look into other ones
    print("metaphone: ", metaphone(query))

    results = forest.search(query)
    pprint(results)
