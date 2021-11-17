""" run.py

File to act as the "driver" for the program.
For now, use this file for quick iteration and running
tiny experiments to get minimum functionality working.
"""

from trie import Trie, Forest
import Levenshtein as lv

if __name__ == "__main__":
    t1 = Trie()
    t1.add_entry("special")
    t1.add_entry("specific")
    t1.add_entry("spice")
    # print(t1.search("spice", 2))

    t2 = Trie()
    t2.add_entry("expose")
    t2.add_entry("external")
    t2.add_entry("extreme")
    t2.add_entry("airport")

    forest = Forest()
    forest.add_trie(t1)
    forest.add_trie(t2)

    results = forest.search("spice", 1)

    # similar_words = t.search(word="airport", max_edit_distance=2)
    # print("search word: airport", similar_words)

    # # NOTE look at this with dumps, should return with ED 1
    # similar_words = t.search(word="aiport", max_edit_distance=2)
    # print("search word: aiport\nedit distance: 2\nresults:", similar_words)

    # similar_words = t.search(word="aiport", max_edit_distance=1)
    # print("search word: aiport\nedit distance: 1\nresults:", similar_words)

    # # NOTE look at this, should return with ED 2
    # similar_words = t.search(word="aipot", max_edit_distance=2)
    # print("search word: aipot\nedit distance: 2\nresults:", similar_words)

    # similar_words = t.search(word="irport", max_edit_distance=2)
    # print("search word: irport", similar_words)

    # similar_words = t.search(word="rport", max_edit_distance=2)
    # print("search word: rport", similar_words)

    # similar_words = t.search(word="port", max_edit_distance=2)
    # print("search word: port", similar_words)
