class TrieNode:
    def __init__(self, char: str):
        self.char = char
        self.is_end = False
        self.counter = 0
        self.children = {}
