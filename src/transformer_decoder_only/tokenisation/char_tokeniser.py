"""
Character-level tokenisation utilities for the educational decoder-only transformer project.

This module is intentionally simple, but it plays a foundational role in the
entire language-modelling pipeline. It gives the rest of the repository a stable
way to talk about:

- how raw text is broken down into tokens
- how each token is mapped to an integer id
- how integer ids are mapped back to text
- how vocabulary size is determined from the corpus

Keep tokenisation logic in one place makes the project easier to extend because:

- `datasets` can focus on slicing token streams into training examples
- `models` can focus on transforming token ids into useful representations
- `training` can focus on optimisation rather than text preprocessing
- `inference` can focus on generation rather than encoding details

In plain language:

- this module answers the question, "How do we turn text into numbers?"
- it also answers the reverse question, "How do we turn model outputs back into text?"

Notes
-----
- This project uses character-level tokenisation for the first implementation.
- That means each individual character becomes one token. For example, the string:

    "hello"

  becomes the token sequence:

    ["h", "e", "l", "l", "o"]

- This is much simpler than word-level or subword tokenisation, which makes it a
  very good educational starting point.

- The trade-off is that sequences become longer, because common words are not
  compressed into larger units. For small pedagogical model, this is an
  acceptable trade.

- We sort the discovery vocabulary before assigning ids.
  This gives us a deterministic mapping from characters to integer ids.

  In practice, that means:

  - repeated runs on the same corpus give the same vocabulary ordering
  - debugging becomes easier
  - tests become easier to write and reason about
"""

from dataclasses import dataclass, field

@dataclass(slots=True)
class CharTokeniser:
    """
    Character-level tokeniser for small educational language-modelling experiments.

    Attributes
    ----------
    stoi : dict[str, int]
        Mapping from string token to integer id.

        The name `stoi` stands for "string to integer".
        In this project, each string token is a single character.
    
    itos : dict[int, str]
        Mapping from integer id back to string token.

        The name `itos` stands for "integer to string".

    vocabulary : list[str]
        Sorted list of all unique characters known to the tokeniser.

    Notes
    -----
    - A tokeniser has two main responsibilities:
        
        1. encode text into token ids
        2. decode token ids back into text

    - For this project, the tokeniser is intentionally small and explicit.
      We do not hide anything behind a third-party abstraction because the goal
      is to understand the full modelling pipeline.

    Example
    -------
    Build a tokeniser from a tiny text corpus.

    >>> tokeniser = CharTokeniser.from_text("hello")
    >>> tokeniser.vocabulary
        ['e', 'h', 'l', 'o']

    This means the tokeniser has discovered four unique characters and assigned
    each of them a stable integer id.
    """

    stoi: dict[str, int] = field(default_factory=dict)
    itos: dict[int, str] = field(default_factory=dict)
    vocabulary: list[str] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> "CharTokeniser":
        """
        Build a character tokeniser directly from raw text.

        Parameters
        ----------
        text : str
            Raw text corpus from which to discover the vocabulary.

        Returns
        -------
        CharTokeniser
            A tokeniser whose vocabulary is built from the unique characters
            founder in `text`.

        Notes
        -----
        The main idea is simple:

        - find the set of unique characters in the corpus
        - sort them to make the mapping deterministic
        - assign each character a unique integer id

        Example
        -------
        Suppose the input text is:

            "banana"

        The unique characters are:

            {"b", "a", "n"}

        After sorting, the vocabulary becomes:

            ["a", "b", "n"]

        So the mappings become:

        - "a" -> 0
        - "b" -> 1
        - "n" -> 2

        In plain language, this method says:

            "Look at the corpus, discover every character we need to know
             about, and build the lookup tables once."
        """
        if not text:
            raise ValueError("Cannot build a CharTokeniser from empty text.")
        
        # We sort the unique characters so that the vocabulary order is stable
        #   - This is very helpful for debugging and testing because the same 
        #     text will always produce the same character-to-id mapping.
        vocabulary = sorted(set(text))

        # Build the forward mapping:
        #
        #   character -> integer id
        #
        # Example:
        #
        #   {"a": 0, "b": 1, "n": 2}
        stoi = {character: index for index, character in enumerate(vocabulary)}

        # Build the reverse mapping:
        #
        #   integer id -> character
        #
        # Example:
        #
        #   {0: "a", 1: "b", 2: "c"}
        itos = {index: character for index, character in enumerate(vocabulary)}

        return cls(stoi=stoi, itos=itos, vocabulary=vocabulary)
    
    @property
    def vocab_size(self) -> int:
        """
        Return the number of unique characters in the tokeniser's vocabulary.

        This is the total number of distinct character tokens that the tokeniser
        currently knows how to map between text and integer token ids.

        Returns
        -------
        int
            The number of unique character tokens in the vocabulary.

        Notes
        -----
        - This value is one of the most important quantities in the whole language
          model pipeline because it determines several key dimensions:
        - Embedding table size:
            the token embedding layer must contain one learned embedding vector for
            each token in the vocabulary. If the vocabulary size is V and the
            embedding dimension is D, then the embedding table has shape
            ``(V, D)``.
        - Final output size:
            at each position, the model produces one logit for every possible token
            in the vocabulary. If the vocabulary size is V, then the model's final
            output dimension is V.
        - Valid token id range:
            token ids must be integers in the range `0` to `V - 1` inclusive.

        Example
        -------
        If the vocabulary contains 40 unique characters, then:

            vocab_size == 40
        
        Valid token ids range from 0 to 39. The model must output 40 logits at each
        prediction step, one for each possible next token.
        """

        return len(self.vocabulary)
    
    def encode(self, text: str) -> list[int]:
        """
        Convert raw text into a list of integer token ids.

        Parameters
        ----------
        text : str
            Input text to encode.

        Returns
        -------
        list[int]
            List of token ids, one per character.

        Notes
        -----
        - Because this is a character-level tokeniser, encoding is straightforward:

            - take each character in the input string
            - look up its integer id in `stoi`
            - return the resulting sequence of integers

        Example
        -------
        Suppose the vocabulary mapping is:

        - "a" -> 0
        - "b" -> 1
        - "c" -> 2

        Then encoding:

            "banana"

        produces:

            [1, 0, 2, 0, 2, 0]

        In plain language, we are replacing each character with its vocabulary index.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        
        # We build the encoded sequence one character at a time
        #   - This allows for errors to be reported clearly is an unknown character
        #     appears.
        token_ids: list[int] = []

        for character in text:
            if character not in self.stoi:
                raise ValueError(
                    f"Character {character!r} is not in the tokeniser vocabulary."
                )
            
            token_ids.append(self.stoi[character])

        return token_ids
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Convert a list of integer token ids back into text.

        Parameters
        ----------
        token_ids : list[int]
            Sequence of integer token ids to decode.

        Returns
        -------
        str
            Decoded text string.

        Notes
        -----
        - Decoding reverses the encoding process:
            - take each integer id
            - look up the corresponding character in `itos`
            - join the characters together into one string

        Example
        -------
        Suppose the reverse mapping is:

        - 0 -> "a"
        - 1 -> "b"
        - 2 -> "c"

        Then decoding:

            [1, 0, 2, 0, 2, 0]

        produces:

            "banana"

        In plain language, decoding lets us inspect model inputs and outputs as
        readable text rather than raw integer sequences.
        """

        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list of integers.")
        
        characters: list[str] = []
        
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("Every token id must be an integer.")
            
            if token_id not in self.itos:
                raise ValueError(
                    f"Token id {token_id!r} is not in the tokeniser vocabulary."
                )
            
            characters.append(self.itos[token_id])

        return "".join(characters)
    
    def encode_text(self, text: str) -> list[int]:
        """
        Alias for `encode`.

        Returns
        -------
        list[int]
            List of token ids, one per character.

        Notes
        -----
        - This method exists mainly for readibility in calling code.
        - Some places in the project may read more clearly as:

            `tokeniser.encode_tokens(text)`

          rather than simply:

            `tokeniser.encode(text)`

          Both methods perform the same operation. 
        """
        return self.encode(text)
    
    def decode_text(self, token_ids: list[int]) -> str:
        """
        Alias for `decode`.

        Returns
        -------
        str
            Decoded text string.

        Notes
        -----
        - This method exists mainly for readibility in calling code.
        - Some places in the project may read more clearly as:

            `tokeniser.decode_tokens(token_ids)`

          rather than simply:

            `tokeniser.decode(token_ids)`

          Both methods perform the same operation. 
        """
        return self.decode(token_ids)
    
    def token_to_id(self, token: str) -> int:
        """
        Return the integer id for a single character token.

        Parameters
        ----------
        token : str
            A single-character token.

        Returns
        -------
        int
            Integer id associated with that token.

        Notes
        -----
        This helper is useful when writing tests, debugging examples, or small
        demonstrations where it is convenient to inspect the mapping explicitly.
        """

        if not isinstance(token, str):
            raise TypeError("token must be a string.")
        
        if len(token) != 1:
            raise ValueError("token must be a single character.")
        
        if token not in self.stoi:
            raise ValueError(
                f"Token {token!r} is not in the tokeniser vocabulary."
            )
        
        return self.stoi[token]
    
    def id_to_token(self, token_id: int) -> str:
        """
        Return the character token for a single integer id.

        Parameters
        ----------
        token_id : int
            Integer token id.

        Returns
        -------
        str
            Character associated with that id.

        Notes
        -----
        - This is the inverse of `token_to_id`.
        - It is the most useful for debugging and small demonstrations.
        """

        if not isinstance(token_id, int):
            raise TypeError("token_id must be an integer.")
        
        if token_id not in self.itos:
            raise ValueError(
                f"Token id {token_id!r} is not in the tokeniser vocabulary."
            )
        
        return self.itos[token_id]
    
    