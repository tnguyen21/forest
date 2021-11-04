""" run.py

File to act as the "driver" for the program.
For now, use this file for quick iteration and running
tiny experiments to get minimum functionality working.
"""

from trie import Trie
import Levenshtein as lv

if __name__ == "__main__":
    t = Trie()
    # t.add_entry("special")
    # t.add_entry("specific")
    # t.add_entry("spice")
    # t.add_entry("expose")
    # t.add_entry("external")
    # t.add_entry("extreme")
    t.add_entry("airport")
    # t.dump()
    # similar_words = t.search(word="airport", max_edit_distance=2)
    # print("search word: airport", similar_words)

    # NOTE look at this with dumps, should return with ED 1
    similar_words = t.search(word="aiport", max_edit_distance=2)
    print("search word: aiport", similar_words)

    # # NOTE look at this, should return with ED 2
    # similar_words = t.search(word="aipot", max_edit_distance=2)
    # print("search word: aipot", similar_words)

    # similar_words = t.search(word="irport", max_edit_distance=2)
    # print("search word: irport", similar_words)

    # similar_words = t.search(word="rport", max_edit_distance=2)
    # print("search word: rport", similar_words)

    # similar_words = t.search(word="port", max_edit_distance=2)
    # print("search word: port", similar_words)
