import unittest
from context import trie


class TestTrieMethods(unittest.TestCase):
    def test_insert(self):
        t = trie.Trie()
        t.insert("bag")
        result = t.query("b")
        self.assertEqual(result, [("bag", 1)])


if __name__ == "__main__":
    unittest.main()
