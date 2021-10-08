class TrieNode:
    def __init__(
        self,
        char: str,
        node_level: int = 0,
        is_end_of_word: bool = False,
        children: dict = {},
        parent: 'TrieNode' = None
    ):
        """
        Constructor for node in Trie

        Args:
            char: character that this trie node represents
            node_level: level of this node in the trie
            is_end_of_word: bool whether this node is the end of a word
            children: dictionary of children nodes
            parent: parent node
        """
        self.char = char
        self.node_level = node_level
        self.is_end_of_word = False
        self.children = {}
        self.parent = None
