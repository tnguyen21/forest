""" util/string_utils.py

File for string utilities (mostly regularizing inputs)
"""


def string_preprocess(string: str) -> str:
    """
    Preprocesses a string to remove all non-alphanumeric characters
    and converts it to lowercase, and trims whitespace.

    Args:
        string: the string to preprocess

    Returns:
        the processed string
    """
    string = string.lower()
    string = string.lstrip().rstrip()
    return "".join(c for c in string if c.isalnum())
