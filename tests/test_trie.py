import unittest
from .context import trie

# TODO add fixture for sample trie that we can search easily


class TestTrieMethods(unittest.TestCase):
    def test_single_char_add_entry(self):
        t = trie.Trie()
        t.add_entry("a")
        self.assertEqual(t.root.children["a"].char, "a")

    def test_multi_char_add_entry(self):
        t = trie.Trie()
        t.add_entry("aa")
        self.assertEqual(t.root.children["a"].char, "a")
        self.assertEqual(t.root.children["a"].children["a"].char, "a")

    def test_multi_word_same_prefix_add_entry(self):
        t = trie.Trie()
        t.add_entry("aa")
        t.add_entry("ab")
        self.assertEqual(t.root.children["a"].char, "a")
        # we expect two children, one to have value 'a', another of value 'b'
        self.assertEqual(t.root.children["a"].children["a"].char, "a")
        self.assertEqual(t.root.children["a"].children["b"].char, "b")

    def test_multi_word_diff_prefix_add_entry(self):
        t = trie.Trie()
        t.add_entry("a")
        t.add_entry("b")
        self.assertEqual(t.root.children["a"].char, "a")
        self.assertEqual(t.root.children["b"].char, "b")


class TestTrieSearch(unittest.TestCase):
    # TODO add test casses for search
    def example_test(self):
        return True


if __name__ == "__main__":
    unittest.main()
