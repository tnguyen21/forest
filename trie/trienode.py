class TrieNode:
    def __init__(
        self,
        char: str,
        is_end_of_word: bool = False,
        children: dict = {},
        parent: 'TrieNode' = None
    ):
        """
        Constructor for node in Trie

        Args:
            char: character that this trie node represents
            is_end_of_word: bool whether this node is the end of a word
            children: dictionary of children nodes
            parent: parent node
        """
        self.char = char
        self.is_end_of_word = False
        self.children = {}
        self.parent = None
