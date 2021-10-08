from trienode import TrieNode


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
        """Insert a word into the trie"""
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node
        node.is_end = True
        node.counter += 1

    def dfs(self, node, prefix):
        """Depth-first traversal of the trie

        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a
                word while traversing the trie
        """
        if node.is_end:
            self.output.append((prefix + node.char, node.counter))

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

    def search(self, word, maxCost):
        """
        Copied from http://stevehanov.ca/blog/index.php?id=114
        Here for testing purposes only
        """
        currentRow = range(len(word) + 1)
        results = []
        # recursively search each branch of the trie
        for letter in self.root.children:
            self.searchRecursive(self.root.children[letter], letter, word, currentRow, results, maxCost) # noqa E501

        return results

    # This recursive helper is used by the search function above. It assumes
    # that the previousRow has been filled in already.
    def searchRecursive(self, node, letter, word, previousRow, results, maxCost): # noqa E501
        """
        Copied from http://stevehanov.ca/blog/index.php?id=114
        Here for testing purposes only
        """
        columns = len(word) + 1
        currentRow = [previousRow[0] + 1]

        # Build one row for the letter, with a column for each letter in
        # the target word, plus one for the empty string at column 0
        for column in range(1, columns):
            insertCost = currentRow[column - 1] + 1
            deleteCost = previousRow[column] + 1

            if word[column - 1] != letter:
                replaceCost = previousRow[column - 1] + 1
            else:
                replaceCost = previousRow[column - 1]

            currentRow.append(min(insertCost, deleteCost, replaceCost))

        # if the last entry in the row indicates the optimal cost is less than
        # the maximum cost, and there is a word in this trie node, then add it.
        if currentRow[-1] <= maxCost and node.word is not None:
            results.append((node.word, currentRow[-1]))
        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min(currentRow) <= maxCost:
            for letter in node.children:
                self.searchRecursive(node.children[letter], letter, word, currentRow, results, maxCost) # noqa E501


if __name__ == "__main__":
    t = Trie()
    t.insert("bag")
    t.insert("brave")
    print(t.query("b"))
