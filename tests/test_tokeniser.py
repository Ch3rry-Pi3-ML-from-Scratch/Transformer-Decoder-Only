"""
Tests for the character-level tokeniser used in the educational transformer project.

This module is intentionally small, but it plays an important role in the build.
It helps us lock in the expected behaviour of the tokeniser before the rest of
the project starts depending on it.

That matters because the tokeniser sits very early in the pipeline:

- raw text is first passed into the tokeniser
- the tokeniser converts text into integer ids
- the dataset will later slice those ids into training examples
- the model will later consume those ids as inputs
- generated token ids will eventually be decoded back into text

If tokenisation is wrong, then everything downstream becomes harder to reason
about. So these tests are designed to check the basics carefully and early.

In plain language:

- does the tokeniser discover the right vocabulary?
- does it encode text correctly?
- does it decode ids correctly?
- does it fail clearly when given invalid input?

Notes
-----
- These tests use `pytest`.
- Each test focuses on one small piece of behaviour.
- The goal is not cleverness. The goal is clarity and confidence.

- We use tiny text examples such as `"banana"` because they are easy to reason
  about by hand.

  For example, if the sorted vocabulary is:

    ["a", "b", "n"]

  then the mapping should be:
  
    "a" -> 0
    "b" -> 1
    "n" -> 2

  and encoding `"banana"` should produce:

    [1, 0, 2, 0, 2, 0]
"""

import pytest

from transformer_decoder_only.tokenisation.char_tokeniser import CharTokeniser

def test_from_text_builds_sorted_vocabulary_and_lookup_tables() -> None:
    """
    Tests that the tokeniser builds a deterministic sorted vocabulary.

    Notes
    -----
    - We deliberately use a tiny repeated-word example here because it makes the
      expected behaviour easy to inspect manually.
    - For the text:

        "banana"

      the unique characters are:

        {"b", "a", "n"}

      and the sorted vocabulary should be:

        ["a", "b", "n"]

      From that, we expect:

        stoi["a"] == 0
        stoi["b"] == 1
        stoi["n"] == 2

      and the inverse mapping should agree.
    """

    tokeniser = CharTokeniser.from_text("banana")

    assert tokeniser.vocabulary == ["a", "b", "n"]
    assert tokeniser.stoi == {"a": 0, "b": 1, "n": 2}
    assert tokeniser.itos == {0: "a", 1: "b", 2: "n"}

def test_vocab_size_matches_number_of_unique_characters() -> None:
    """
    Test that `vocab_size` reports the number of unique characters.

    This is important because the rest of the model will later depend on
    `vocab_size` when creating:

    - the token embedding table
    - the final output projection over the vocabulary
    """

    tokeniser = CharTokeniser.from_text("hello")

    # The unique characters are: "e", "h", "l", "o"
    #   - The vocabulary size should be 4.
    assert tokeniser.vocab_size == 4

def test_encode_converts_text_to_expected_token_ids() -> None:
    """
    Test that encoding follows the discovered `stoi` mapping correctly.

    For the text corpus `"banana"`, the sorted vocabulary is:

        "a" -> 0
        "b" -> 1
        "n" -> 2

    So encoding `"banana"` should give us:

        [1, 0, 2, 0, 2, 0]
    """
    tokeniser = CharTokeniser.from_text("banana")

    encoded = tokeniser.encode("banana")

    assert encoded == [1, 0, 2, 0, 2, 0]

def test_decode_converts_token_ids_back_to_text() -> None:
    """
    Test that decoding reverses the tokenisation process correctly.

    This is the inverse of encoding:

    - ids go in
    - readable text comes out
    """

    tokeniser = CharTokeniser.from_text("banana")

    decoded = tokeniser.decode([1, 0, 2, 0, 2, 0])

    assert decoded == "banana"

def test_encode_and_decode_form_a_round_trip_for_known_text() -> None:
    """
    Test the most important practical tokeniser property: round-tripping.

    If the input text contains only known characters, then:

    - encode it
    - decode the resulting ids

    and we should get back the original text unchanged.

    This is a very useful sanity check because it confirms that:

    - the forwards mapping is correct
    - the reverse mapping is correct
    - the two mappings are consistent with one another
    """

    original_text = "hello"
    tokeniser = CharTokeniser.from_text(original_text)

    encode = tokeniser.encode(original_text)
    decode = tokeniser.decode(encode)

    assert decode == original_text

def test_encode_text_alias_matches_encode() -> None:
    """
    Test that the readability alias `encode_text` behaves exactly like `encode`.

    This is not mathematically deep, but it is worth locking in because later
    calling code may use either form.
    """

    tokeniser = CharTokeniser.from_text("hello")

    assert tokeniser.encode_text("hello") == tokeniser.encode("hello")

def test_decode_tokens_alias_matches_decode() -> None:
    """
    Test that the readability alias `decode_tokens` behaves exactly like `decode`.

    Again, this is a small interface test rather than a modelling test, but it
    makes the public API clearer and safer to refactor later.
    """

    tokeniser = CharTokeniser.from_text("hello")
    encoded = tokeniser.encode("hello")

    assert tokeniser.decode_text(encoded) == tokeniser.decode(encoded)

def test_token_to_id_returns_expected_integer() -> None:
    """
    Test the convenience helper for inspecting a single token mapping.

    These helper methods are mainly useful for:

    - tests
    - debugging
    - teaching and demonstration
    """

    tokeniser = CharTokeniser.from_text("banana")

    assert tokeniser.token_to_id("b") == 1
    assert tokeniser.token_to_id("a") == 0
    assert tokeniser.token_to_id("n") == 2

def test_id_to_token_returns_expected_character() -> None:
    """
    Test the reverse convenience helper for inspecting a single id mapping.
    """

    tokeniser = CharTokeniser.from_text("banana")

    assert tokeniser.id_to_token(0) == "a"
    assert tokeniser.id_to_token(1) == "b"
    assert tokeniser.id_to_token(2) == "n"

def test_from_text_raises_error_for_empty_text() -> None:
    """
    Test that building a tokeniser from empty text fails clearly.

    This is a helpful guard because an empty corpus cannot define a useful
    vocabulary.
    """

    with pytest.raises(ValueError, match="empty text"):
        CharTokeniser.from_text("")

def test_encode_raises_error_for_unknown_character() -> None:
    """
    Test that encoding fails clearly when the input contains unseen characters.

    This matters because the tokeniser vocabulary is built from the training
    corpus. If a later prompt includes a character that never appeared in that
    corpus, the tokeniser should complain explicitly rather than silently doing
    something confusing.
    """

    tokeniser = CharTokeniser.from_text("abc")

    with pytest.raises(ValueError, match="not in the tokeniser vocabulary"):
        tokeniser.encode("abd")

def test_encode_raises_type_error_for_non_string_input() -> None:
    """
    Test that `encode` rejects non-string input with a clear error.

    The tokeniser is designed to work on Python strings, not arbitrary objects.
    """

    tokeniser = CharTokeniser.from_text("abc")

    with pytest.raises(TypeError, match="text must be a string"):
        tokeniser.encode(123) # type: ignore[arg-type]

def test_decode_raises_type_error_when_input_is_not_a_list() -> None:
    """
    Test that `decode` expects a list of integer token ids.

    This keeps the interface explicit and beginner-friendly.
    """

    tokeniser = CharTokeniser.from_text("abc")

    with pytest.raises(TypeError, match="token_ids must be a list of integers"):
        tokeniser.decode("012") # type: ignore[arg-type]

def test_decode_raises_type_error_when_any_token_id_is_not_an_integer() -> None:
    """
    Test that every decoded token id must be an integer.

    This guards against mixed-type input such as `[0, "1", 2]`.
    """

    tokeniser = CharTokeniser.from_text("abc")

    with pytest.raises(TypeError, match="Every token id must be an integer"):
        tokeniser.decode([0, "1", 2]) # type: ignore[arg-type]

def test_decode_raises_error_for_unknown_token_id() -> None:
    """
    Test that decoding fails clearly when an id is outside the known vocabulary.
    """

    tokeniser = CharTokeniser.from_text("abc")

    with pytest.raises(ValueError, match="not in the tokeniser vocabulary"):
        tokeniser.decode([0, 1, 99])

def test_token_to_id_raises_for_invalid_token_length() -> None:
    """
    Test that `token_to_id` expects exactly one character.

    Because this is a character-level tokeniser, a token should not be a whole
    word or an empty string.
    """

    tokeniser = CharTokeniser.from_text("abc")

    with pytest.raises(ValueError, match="single character"):
        tokeniser.token_to_id("ab")

def test_id_to_token_raises_for_unknown_id() -> None:
    """
    Test that `id_to_token` rejects ids outside the known mapping.
    """

    tokeniser = CharTokeniser.from_text("abc")

    with pytest.raises(ValueError, match="not in the tokeniser vocabulary"):
        tokeniser.id_to_token(5901)