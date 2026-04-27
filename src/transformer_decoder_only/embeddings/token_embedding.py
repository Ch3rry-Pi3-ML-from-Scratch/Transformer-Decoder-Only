"""
Token-embedding module for the educational decoder-only transformer project.

This module is intentionally small, but it introduces one of the most important
ideas in the whole module. It gives the rest of the repository a stable way to
talk about:

- how integer tokens ids become dense vectors
- how symbolic tokens enter the neural network
- how vocabulary items receive learned representations
- how the model's hidden space begins

Keeping token-embedding logic in one place makes the project easier to extend because:

- `tokenisation` can focus on text-to-id conversion
- `datasets` can focus on next-token example construction
- `embeddings` can focus on representation building
- `attention` can later operate on meaninful dense vectors rather than raw ids
- `models` can compose this embedding stage with positional information and decoder blocks

In plain language:

- this module answers the question, 

    "How does the model read a token id?"

- it turns each integer token index into a learned vector of numbers

Notes
-----
- Neural networks do not work directly with symbolic categories such as characters.
  For example, the token id:

    17

  does not, by itself, tell the model anything meaningful about similarity, context,
  or usage. It is just an index.
- A token embedding solves that problem by learning a vector for each token in
  the vocabulary.
- If:

    `vocab_size = V`
    `embedding_dim = C`

  then the embedding table has shape:

    `(V, C)`

- You can think of this as one learned row vector per token in the vocabulary.
  For example:

    - row 0 stores the vector for token 0
    - row 1 stores the vector for token 1
    - row 2 stores the vector for token 2

- When the model receives a batch of token ids with shape:

    `(B, T)`

  the embedding layer performs a table lookup and returns:

    `(B, T, C)`

  where:

    - `B` is the batch size
    - `T` is the sequence length
    - `C` is the embedding dimension

- This is the first step where the model moves from discrete symbolic input
  into a continuous learned representation space.
"""

import torch
from torch import Tensor
from torch import nn

from transformer_decoder_only.config.default import ModelConfig
from transformer_decoder_only.utils.shapes import assert_rank, format_shape

class TokenEmbedding(nn.Module):
    """
    Learned token-embedding layer for a character-level language model.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing at least:

        - `vocab_size`
        - `embedding_dim`

    Attributes
    ----------
    vocab_size : int
        Number of unique tokens in the vocabulary.
    embedding_dim : int
        Width of the learned vector representation for each token.
    embedding : nn.Embedding
        PyTorch embedding table that stores one learned vector per token id.

    Notes
    -----
    - The conceptual job of this module is simple:

        - input: integer token ids
        - output: learned dense vectors
    
    - If the model sees the token id sequence:

        [4, 1, 9]

      it does not treat those numbers as numeric values in the usual sense.
      Instead, it uses them as row indices into the embedding table.

    - In plain language, this module says:

        "For each token id, fetch the learned vector associated with that token."

    Tensor shapes
    -------------
    Let:

    - `B` = batch size
    - `T` = sequence length
    - `V` = vocabulary size
    - `C` = embedding dimension

    Then:

    - embedding table shape: `(V, C)`
    - input token id shape: `(B, T)`
    - output embedding shape: `(B, T, C)`

    Example
    -------
    Suppose a tiny character-level tokeniser knows the following vocabulary:

        "a" -> 0
        "b" -> 1
        "c" -> 2
        " " -> 3

    then:

        vocab_size = 4

    If:

        embedding_dim = 3

    then the embedding table has shape:

        (4, 3)
    
    This means there are 4 learned row vectors, one for each token id, and each
    row has width 3.

    If the input text is:

        "cab"

    then the token id sequence is:

        [2, 0, 1]

    If this sequence is passed as a batch containing one example, the input token
    id tansor has shape:

        (1, 3)

    where:

    - `1` is the batch size
    - `3` is the sequence length

    After the embedding lookup, each token id is replaced by its learned vector,
    so the output embedding tensor has shape:

        (1, 3, 3)

    More generally, if:

        vocab_size = V
        embedding_dim = C
        batch_size - B

    then:

    - embedding table shape = `(V, C)`
    - input token id shape = `(B, T)`
    - output embedding shape = `(B, E, C)`
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialise the token-embedding module.

        Parameters
        ----------
        config : ModelConfig
            Model configuration describing the embedding table shape.

        Notes
        -----
        - We validate the configuration immediately so that invalid settings fail
          early and clearly.
        - In particular, token embeddings require:

            - a positive vocabulary size
            - a positive embedding dimension
        """

        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError("config must be an instance of ModelConfig.")
        
        config.validate()

        if config.vocab_size <= 0:
            raise ValueError(
                "config.vocab_size must be greater than 0 before creating "
                "the token embedding layer."
            )
        
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim

        # Create the learnable embedding table
        #
        # Shape:
        #   
        #   (vocab_size, embedding_dim)
        #
        # Each row corresponds to one token id in the vocabulary
        #   - When a token id appears in the input, PyTorch looks up the
        #     corresponding row vector.
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Convert token ids into learned dense vectors.

        Parameters
        ----------
        token_ids : Tensor
            Integer token ids with shape `(B, T)`.

            Here:

            - `B` is batch size
            - `T` is sequence length

        Return
        ------
        Tensor
            Token embeddings with shape `(B, T, C)` where:

            - `B` is batch size
            - `T` is sequence length
            - `C` is embedding dimension

        Notes
        -----
        - This method performs a simple but very important operation:
            - each integer token id is used as an index
            - the corresponding learned row is fetched from the embedding table
            - the output is a dense tensor of vectors rather than raw ids

        Example
        -------
        Suppose:

        - input token ids have shape `(16, 64)`
        - embedding dimension is `128`

        Then the output has shape:

            (16, 64, 128)

        In plain language:

        - each of the 16 sequences
        - each of the 64 token positions
        - now has a 128-dimensional learned vector at every position
        """

        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor.")
        
        # Token ideas for a batch of sequences should have shape `(B, T)`.
        #   - If a tensor of the wrong rank arrives here, it usually means a bug in:
        #
        #       - the dataset
        #       - the batching logic
        #       - or the calling model code
        assert_rank(token_ids, expected_rank=2, tensor_name="token_ids")

        # Embedding loopup requires integer indices
        #   - In PyTorch, token ids are normally stored as `torch.long`
        if token_ids.dtype != torch.long:
            raise TypeError(
                "token ids must have dtype torch.long for embedding lookup, "
                f"but received dtype {token_ids.dtype}."
            )
        
        # It is also useful to guard against impossible token ids early
        #   - Valid token ids must lie in:
        #
        #       [0, vocab_size - 1]
        #
        #     If not, the embedding lookup would fail or behave unexpectedly.
        if torch.any(token_ids < 0):
            raise ValueError("token_ids must not contain negative values.")
        
        if torch.any(token_ids >= self.vocab_size):
            raise ValueError(
                f"token_ids contain values outside the vocabulary range "
                f"[0, {self.vocab_size - 1}]."
            )
        
        # Perform the embedding-table lookup
        #
        # Input shape:
        #   token_ids -> (B, T)
        #
        # Output shape:
        #   embeddings -> (B, T, C)
        #
        # where:
        #   B = batch size
        #   T = sequence length
        #   C = embedding dimension
        embeddings = self.embedding(token_ids)

        # This check is slightly redundant because `nn.Embedding` is reliable,
        # but it makes the intended output shape explicit and gives clearer
        # messages during development if something upstream changes.
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                "Token embedding output has the wrong final dimension. "
                f"Expected {self.embedding_dim}, received shape {format_shape(embeddings)}."
            )

        return embeddings