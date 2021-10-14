from .trienode import TrieNode
from utils.string_utils import string_preprocess


class Trie:
    def __init__(
        self,
        root: TrieNode = TrieNode("", node_level=0),
        entry_list: [str] = [],
        max_current_search_level: int = 0,
    ):
        self.root = root
        self.entry_list = entry_list
        self.max_current_search_level = max_current_search_level

    def insert(self, word):
        """
        Insert a word into the trie

        Args:
            word: the word to be inserted
        """
        node = self.root
        node_level = 0
        word = string_preprocess(word)
        for char in word:
            node_level += 1
            if char in node.children:
                node = node.children[char]
            else:
                # quirk of impl, need to name parent argument
                # since in constructor we can also pass in
                # is_end_of_word and dict of children during contruction
                new_node = TrieNode(char, node_level, parent=node)
                node.children[char] = new_node
                node = new_node
        # out of for loop, end of word is reached
        node.is_end_of_word = True

    def dfs(self, node, prefix):
        """Depth-first traversal of the trie

        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a
                word while traversing the trie
        """
        if node.is_end_of_word:
            self.output.append((prefix + node.char, node.node_level))

        for child in node.children.values():
            self.dfs(child, prefix + node.char)

    def query(self, x):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of
        times they have been inserted
        """
        self.output = []
        node = self.root

        # Check if the prefix is in the trie
        for char in x:
            if char in node.children:
                node = node.children[char]
            else:
                # cannot found the prefix, return empty list
                return []

        # Traverse the trie to get all candidates
        self.dfs(node, x[:-1])

        # Sort the results in reverse order and return
        return sorted(self.output, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    t = Trie()
    t.insert("bag")
    t.insert("brave")
    print(t.query("b"))
