""" trie/trie.py

File which holds the Trie class and associated
algorithms.
"""
from typing import List, Tuple
from .trienode import TrieNode
from Levenshtein import jaro_winkler


class Trie:
    def __init__(self, max_edit_distance: int = 2, min_jaro_similarity: float = 0.0):
        self.root = TrieNode("", self)
        self.entry_list = []  # think about removing this
        self.max_current_search_level = 0
        self.max_edit_distance = max_edit_distance
        self.min_jaro_similarity = min_jaro_similarity
        self.active_nodes = {}
        self.max_depth = 0

    def dump(self):
        """
        Print trie into a formatted string
        """
        self.root.dump()

    def add_entry(self, word) -> TrieNode:
        """
        Insert a word into the trie

        Args:
            word: the word to be inserted
        """
        # add to entry list, check for duplicates
        if word not in self.entry_list:
            self.entry_list.append(word)

        # return last node added (pointer to last node)
        # TODO we want to connect this last node to entry list
        end_node = self.root.add_entry(word)

        return end_node

    def dfs(self, node, prefix):
        """
        Depth-first traversal of the trie

        Args:
            node: the node to start with
            prefix: the current prefix, for tracing a
                    word while traversing the trie
        """
        if node.is_end_of_word:
            self.output.append((prefix + node.char, node.node_level))

        for child in node.children.values():
            self.dfs(child, prefix + node.char)

    def query(self, x):
        """
        Given an input (a prefix), retrieve all words stored in
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

    def is_active_nodes_empty(self):
        """
        Check if active nodes is empty
        """
        active_nodes_empty = True
        for level in range(self.max_depth + 1):
            if len(self.active_nodes[level]) > 0:
                active_nodes_empty = False
                break

        return active_nodes_empty

    def is_all_active_nodes_end_of_words(self):
        """
        Check if all active nodes are end of words
        """
        all_active_nodes_end_of_words = True
        for level in range(self.max_depth + 1):
            for node in self.active_nodes[level]:
                if not node.is_end_of_word:
                    all_active_nodes_end_of_words = False
                    break

        return all_active_nodes_end_of_words

    def update_further_children(self, node: "TrieNode", char: str) -> None:
        """
        Update further children of the node recursively until
        max_edit_distance is hit or the end of the trie is reached
        Args:
            node: the node to start with
            char: the current char being searched with
        """
        for child_char in node.children:
            child_node = node.children[child_char]

            if child_node in self.active_nodes.get(node.node_level + 1, []):
                tmp_ed = child_node.edit_distance
            else:
                tmp_ed = self.max_edit_distance + 1

            # if children don't match, increase ED for those nodes
            # take the smaller of this ED or the ED
            if child_char != char:
                child_node.edit_distance = min(node.edit_distance + 1, tmp_ed)
            else:
                child_node.edit_distance = min(node.edit_distance, tmp_ed)

            if (child_node.edit_distance <= self.max_edit_distance) and (
                child_node not in self.active_nodes.get(node.node_level + 1, [])
            ):
                self.active_nodes[node.node_level + 1].append(child_node)

            # keep matching child's children (i.e. down the trie)
            # until max_edit_distance is hit or the end of the trie is reached
            if child_node.edit_distance <= self.max_edit_distance:
                self.update_further_children(child_node, "")
        return

    def search(
        self, word: str, max_edit_distance: int = -1
    ) -> List[Tuple[str, int, float]]:
        """
        Given a word, conduct a similarity search on the trie using the
        notion of an "edit distance" and keeping track of how edit distance
        changes as we traverse the trie.

        Args:
            word: the prefix to search for
            max_edit_distance: the maximum edit distance to allow
        Returns:
            A list of words in the trie that are within the edit distance
            of the word. with their edit distance
        """
        if max_edit_distance < 0:
            max_edit_distance = self.max_edit_distance
        else:
            self.max_edit_distance = max_edit_distance
        # init lists of active nodes
        self.active_nodes = {}

        # initialize lists for each level in active nodes
        for level in range(self.max_depth + 1):
            self.active_nodes[level] = []

        self.root.search_reset(self.max_edit_distance)

        for char in word:
            if self.is_active_nodes_empty():
                break

            for level in range(self.max_depth, -1, -1):
                # want to work from highest ED towards lowest ED
                # ? can we just create this list with valid ED's
                for node in self.active_nodes[level]:

                    # look at all children of the node and then
                    # attempt to move down the trie until we hit the
                    # end of the trie or the max_edit_distance is hit
                    self.update_further_children(node, char)

                    # pointer "stays still" -- we always increment
                    # ? here we can check that if increase ED brings over
                    # ? max of ED, if it does we should rm this from active_nodes
                    node.edit_distance += 1

            # clean up nodes where ED's are > self.max_edit_distance
            # NOTE ideally, we rm nodes the instant EDs cross threshold
            for i in range(self.max_depth + 1):
                filtered_nodes = [
                    n
                    for n in self.active_nodes[i]
                    if n.edit_distance <= self.max_edit_distance
                ]
                self.active_nodes[i] = filtered_nodes

        output = []

        for level in self.active_nodes:
            for node in self.active_nodes[level]:
                # jaro_winkler calculates _similarity_ between 2 strings
                # * by default min_jaro_sim = 0; so if we don't use a threshold
                # * this just slows down our search
                jaro_winkler_similarity = jaro_winkler(node.get_word(), word)
                if (
                    (node.is_end_of_word)
                    and (node.edit_distance <= self.max_edit_distance)
                    and (jaro_winkler_similarity > self.min_jaro_similarity)
                ):
                    output.append(
                        (
                            node.get_word(),
                            node.edit_distance,
                            round(jaro_winkler_similarity, 4),
                        )
                    )

        return output
