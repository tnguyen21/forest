""" run.py

File to act as the "driver" for the program.
For now, use this file for quick iteration and running
tiny experiments to get minimum functionality working.
"""

from trie import Trie

if __name__ == "__main__":
    t = Trie()
    t.insert("bag")
    t.insert("brave")
    # print(t.entry_list)
    # print(t.root.char)
    # print(t.root.children)
    # print(t.root.children['b'].children)
    print(t.query("b"))
