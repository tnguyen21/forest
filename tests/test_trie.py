import unittest
from .context import trie


class TestTrieMethods(unittest.TestCase):
    def test_single_char_insert(self):
        t = trie.Trie()
        t.insert("a")
        self.assertEqual(t.root.children["a"].char, "a")

    def test_multi_char_insert(self):
        t = trie.Trie()
        t.insert("aa")
        self.assertEqual(t.root.children["a"].char, "a")
        self.assertEqual(t.root.children["a"].children["a"].char, "a")

    def test_multi_word_same_prefix_insert(self):
        t = trie.Trie()
        t.insert("aa")
        t.insert("ab")
        self.assertEqual(t.root.children["a"].char, "a")
        # we expect two children, one to have value 'a', another of value 'b'
        self.assertEqual(t.root.children["a"].children["a"].char, "a")
        self.assertEqual(t.root.children["a"].children["b"].char, "b")

    def test_multi_word_diff_prefix_insert(self):
        t = trie.Trie()
        t.insert("a")
        t.insert("b")
        self.assertEqual(t.root.children["a"].char, "a")
        self.assertEqual(t.root.children["b"].char, "b")


if __name__ == "__main__":
    unittest.main()
