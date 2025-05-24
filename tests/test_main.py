import os
import sys
import pytest  # Third-party import
from main import sanitize_input  # Local application import (will be at top)

# Ensure src is in path for imports if tests are run from root
# This needs to be before importing 'main' for runtime, but after for flake8 E402
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


@pytest.mark.parametrize("test_input,expected", [
    ("Hello World", "Hello World"),
    ("<script>alert('XSS')</script>", "&lt;script&gt;alert('XSS')&lt;/script&gt;"),
    ("Some text with !@#$%^&*()", "Some text with !@#$%^&*()"),
    ("  leading and trailing spaces  ", "  leading and trailing spaces  "),
    ("Text with\nnewline", "Text with\nnewline"),  # The regex allows \s, which includes newline
    ("Text with weird chars <>'\"`", "&lt;&gt;'\"`"),  # ` is not in the regex allowed list
    (123, ""),  # Test non-string input
    (None, ""),  # Test None input
])
def test_sanitize_input(test_input, expected):
    assert sanitize_input(test_input) == expected


def test_sanitize_input_removes_disallowed_chars():
    # Test specific removal of characters not in the whitelist
    # The whitelist is r'[^\w\s.,;?!@#$%^&*()-=+\[\]{}:"'/\<>]'
    # So, characters like backtick (`), tilde (~) etc. should be removed if not part of \w or \s
    input_str = "Text with `backtick` and ~tilde~ and |pipe|"
    # Expected: "Text with backtick and tilde and pipe" (assuming `~` and `|` are removed)
    # Actually, \w includes alphanumeric and underscore.
    # The regex removes anything NOT in \w \s . , ; ? ! @ # $ % ^ & * ( ) - = + [ ] { } : " ' / \ < >
    # So ` < > ' " are escaped by html.escape first.
    # Let's test the regex part:
    # After escape: "Text with `backtick` and ~tilde~ and |pipe|" (no change for these)
    # Then regex:
    # ` is NOT in \w etc. -> removed
    # ~ is NOT in \w etc. -> removed
    # | is NOT in \w etc. -> removed
    expected_str = "Text with backtick and tilde and pipe"
    assert sanitize_input(input_str) == expected_str
