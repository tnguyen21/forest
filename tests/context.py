import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E501

import trie  # noqa: F401, E402
