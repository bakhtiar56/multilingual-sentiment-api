"""Tests for the text preprocessing utilities."""

from src.data.preprocess import clean_text


def test_clean_text_removes_urls():
    text = "Check out https://example.com for more info"
    result = clean_text(text)
    assert "https" not in result
    assert "example.com" not in result


def test_clean_text_removes_special_chars():
    text = "Hello!!! How are you???"
    result = clean_text(text)
    assert "!" not in result
    assert "?" not in result


def test_clean_text_normalises_whitespace():
    text = "Too   many    spaces"
    result = clean_text(text)
    assert "  " not in result
    assert result == "Too many spaces"


def test_clean_text_strips_leading_trailing():
    text = "   hello world   "
    result = clean_text(text)
    assert result == result.strip()


def test_clean_text_handles_empty():
    result = clean_text("")
    assert result == ""
