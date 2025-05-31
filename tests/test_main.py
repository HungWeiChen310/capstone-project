import os
import sys
import pytest  # Third-party import
from main import sanitize_input  # Local application import (will be at top)

# Ensure src is in path for imports if tests are run from root
# This needs to be before importing 'main' for runtime, but after for flake8 E402
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # Basic valid inputs
        ("Hello World", "Hello World"),
        ("Some text with !@#$%^&*()", "Some text with !@#$%^&*()"),
        ("  leading and trailing spaces  ", "  leading and trailing spaces  "),
        ("Text with\nnewline", "Text with\nnewline"), # Regex allows \s (includes newline)
        ("Text with numbers 12345", "Text with numbers 12345"),
        ("Text with . , ; ? ! : ' \" / \\", "Text with . , ; ? ! : ' \" / \\"), # Allowed punctuation
        ("Text with []{}()", "Text with []{}()"), # Allowed brackets

        # HTML escaping
        ("<script>alert('XSS')</script>", "&lt;script&gt;alert('XSS')&lt;/script&gt;"),
        ("Test <tag>&'\"", "Test &lt;tag&gt;&amp;'\""), # & is escaped to &amp;
        ("Text with > and < signs", "Text with &gt; and &lt; signs"),
        # Already escaped HTML (html.escape will escape the ampersand again)
        ("&lt;div&gt;", "&amp;lt;div&amp;gt;"),

        # Character removal by regex (after html.escape)
        # ` ~ | are not in the whitelist \w\s.,;?!@#$%^&*()-=+\[\]{}:"'/\<>
        ("Text with `backtick`", "Text with backtick"),
        ("Text with ~tilde~", "Text with tilde"),
        ("Text with |pipe|", "Text with pipe"),
        ("Hello `~World`!|", "Hello World!"), # Mixed

        # Non-string inputs
        (123, ""),
        (None, ""),
        (True, ""), # Boolean, should be treated as non-string
        ([], ""),   # List, should be treated as non-string

        # Empty and whitespace strings
        ("", ""),
        ("   ", "   "), # Only spaces, should remain as \s is allowed

        # Test combined escaping and regex removal
        ("<tag>`test`</tag>", "&lt;tag&gt;test&lt;/tag&gt;"), # `<tag>` becomes `&lt;tag&gt;`, backticks removed
        ("`&`", "&amp;"), # ` removed, & escaped
    ],
)
def test_sanitize_input(test_input, expected):
    """Comprehensive tests for sanitize_input function."""
    assert sanitize_input(test_input) == expected
