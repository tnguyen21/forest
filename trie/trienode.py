""" trie/trienode.py

File which contains the TrieNode class.

TrieNode is a helper for Trie, encapsulating the necessary
data that nodes need to store to make a trie and related
novelty algorithms work.
"""


class TrieNode:
    def __init__(
        self,
        char: str,
        trie: "Trie",
        parent: "TrieNode" = None,
    ):
        """
        Constructor for node in Trie

        Args:
            char: character that this trie node represents
            node_level: level of this node in the trie
            is_end_of_word: bool whether this node is the end of a word
            children: dictionary of children nodes {key: str, value: TrieNode}
            parent: parent node
        """
        self.char = char  # dont know if size of tree will be character, this might take too much memory
        self.node_level = parent.node_level + 1 if parent else 0
        self.is_end_of_word = False
        self.parent = parent  # not usual, might prove useful for an algo impl later, but may also remove
        self.children = {}
        self.edit_distance = 0
        self.word = ""  # consider keeping this, store word in node, can take out parent and is_end_of_word
        self.trie = trie

    def add_entry(self, word: str):
        """
        Recursively add word to a trie

        Args:
            word: word to add to trie

        Return:
            Node representing the end of the word
        """
        if word == "":
            self.is_end_of_word = True
            # keep track of depth of tree for search later
            if self.trie.max_depth < self.node_level:
                self.trie.max_depth = self.node_level
            return self
        else:
            if word[0] in self.children:
                return self.children[word[0]].add_entry(word[1:])
            else:
                self.children[word[0]] = TrieNode(word[0], self.trie, self)
                return self.children[word[0]].add_entry(word[1:])
            return self

    def search_reset(self, max_edit_distance: int):
        """
        Reset the search to the root of the trie
        """
        # initialize edit distance
        self.edit_distance = self.node_level
        
        # include node in list of active nodes
        self.trie.active_nodes[self.node_level].append(self)

        # initialize children if within edit distance level
        if self.node_level < max_edit_distance:
            for child in self.children:
                self.children[child].search_reset(max_edit_distance)
        
        return

    def dump(self):
        """

        Dump the contents of the trie to a string
        """
        if self in self.trie.active_nodes.get(self.node_level):
            print(" " * (self.node_level + 1) * 3, self.char, "*", self.edit_distance)
        else:
            print(" " * (self.node_level + 1) * 3, self.char)
        for child in self.children:
            self.children[child].dump()
