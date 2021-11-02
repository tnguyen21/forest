""" run.py

File to act as the "driver" for the program.
For now, use this file for quick iteration and running
tiny experiments to get minimum functionality working.
"""

from trie import Trie
import Levenshtein as lv

if __name__ == "__main__":
    t = Trie()
    t.add_entry("bag")
    t.add_entry("brave")
    t.add_entry("baker")
    t.add_entry("bake")
    t.add_entry("egg")
    # print(t.entry_list)
    # print(t.root.char)
    # print(t.root.children)
    # print(t.root.children["b"].children)
    # print(t.query("b"))
    # t.similarity_search("b", 2)
    t.search("")
    t.dump()
