"""
Positional-embedding module for the educational decoder-only transformer project.

This module is intentionally small, but it introduces one of the most important
ideas in the transformer architecture. It gives the rest of the repository a
stable way to talk about:

- how sequence order is represented
- how the model distinguishes earlier positions from later ones
- how token meaning and token position are combined
- how a transformer avoids treating a sequence as an unordered bag of tokens

Keeping positional-embedding logic in one place makes the project easier to
extend because:

- `embeddings` can focus on representation building
- `attention` can assume that order information has already been injected
- `models` can compose token and positional information cleanly
- `tests` can verify shape and range assumptions in one place

In plain language:

- this module answers the question,

    "How does the model know where a token is?"

- it gives each sequence position its own learned vector

Core idea
---------
A token embedding tells the model *what* the token is.

A positional embedding tells the model *where* the token is.

Both are needed because language depends on order.

For example, the sequences:

    "abc"
    "cba"

contain the same characters, but in a different order, so they mean different
things structurally. If we used only token embeddings, the model could know
which characters are present, but it would not have a built-in way to represent
their positions.

A positional embedding solves that problem by assigning each position its own
learned vector.

Shape intuition
---------------
Let:

    `T_max` = maximum context length (total number of positions)
    `C`     = embedding dimension
    `B`     = batch size
    `T`     = actual sequence length in the current batch, where `T <= T_max`

Then:

- the position embedding table has shape `(T_max, C)`
- position indices usually have shape `(T,)`
- selected positional embeddings usually have shape `(T, C)`
- after adding a batch dimension for broadcasting, they may have shape `(1, T, C)`
    - The 1 basically means:

        "one shared positional tensor that can be copied conceptually across every
         batch item"

If token embeddings have shape:

    `(B, T, C)`

and positional embeddings have shape:

    `(1, T, C)`

then their elementwise sum has shape:

    `(B, T, C)`

because the positional embeddings are broadcast across the batch dimension.

What the table stores
---------------------
The positional embedding table contains one learned row vector per position.

For example:

- row 0 stores the learned vector for position 0
- row 1 stores the learned vector for position 1
- row 2 stores the learned vector for position 2

and so on, up to:

- row `context_length - 1`

This means that position 0, position 1, and position 2 can all contribute
different information, even if the token at those positions is the same.

Example
-------
Suppose:

    - embedding_dim = 3
    - sequence length `T = 3`

Assume the token embeddings for one sequence are:

    [
        [1,0, 0.2, 0.1],    # token at position 0
        [0.3, 0.8, 0.4],    # token at position 1
        [0.6, 0.1, 0.9],    # token at position 2
    ]

Assume `T_max = 5`, then the positional embeddings table may be:

    [
        [0.1, 0.0, 0.0],   # position 0
        [0.0, 0.1, 0.0],   # position 1
        [0.0, 0.0, 0.1],   # position 2
        [2.0, 0.0, 0.0],   # position 3
        [0.0, 2.0, 0.0],   # position 4 
    ]

Notice here that the actual sequence is only 3 tokens, which is smaller
than the maximum context length of 5. So, we only need the position
embeddings for the first 3 positions from the positions embeddings table.

After elementwise addition, the combined embeddings become:

    [
        [1.1, 0.2, 0.1],   # token meaning + position 0 information
        [0.3, 0.9, 0.4],   # token meaning + position 1 information
        [0.6, 0.1, 1.0],   # token meaning + position 2 information
    ]

So the model now receives vectors that encode both:

- token identity
- token position

Why this matters for attention
------------------------------
Self-attention compares token representations with one another. If those
representations contained only token identity, the model would have a much
weaker sense of sequence order. By adding positional information before the
attention blocks, the model can learn patterns such as:

- "look at the previous token"
- "pay attention to early positions"
- "treat the start of the sequence differently from the end"

In short, positional embeddings inject order information into the model before
attention begins.
"""

import torch
from torch import Tensor
from torch import nn

from transformer_decoder_only.config.default import ModelConfig
from transformer_decoder_only.utils.shapes import assert_rank, format_shape

class PositionEmbedding(nn.Module):
    """
    Learned positional-embedding layer for a decoder-only transformer.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing at least:

        - `context_length`
        - `embedding_dim`

    Attributes
    ----------
    context_length : int
        Maximum supported sequence length.
    embedding_dim : int
        Width of each positional embedding vector.
    embedding : nn.Embedding
        Learnable lookup table storing one vector per position index.

    Notes
    -----
    - The conceptual job of this module is:

        input: sequence positions
        output: learned vectors that represent those positions

    - In this implementation, the forward method receives token ids with shape
      `(B, T)`, uses them only to infer the sequence length `T`, and then creates
      position indices:

        [0, 1, 2, ..., T - 1]

    - These indices are then looked up in the positional embedding table.

    Tensor shapes
    -------------
    Let:

    - `B` = batch size
    - `T` = sequence length
    - `C` = embedding dimension
    - `T_max` = maximum context length

    Then:

    - positional embedding table shape: `(T_max, C)`
    - input token-id shape: `(B, T)`
    - raw looked-up position embedding shape: `(T, C)`
    - returned position embedding shape: `(1, T, C)`

    The leading dimension is `1` rather than `B` because positions are the same
    for every sequence in the batch. PyTorch broadcasting will later expand this
    automatically when we add the positional embeddings to token embeddings.

    Example
    -------
    If:

    - `context_length = 64`
    - `embedding_dim = 128`

    then the positional embedding table has shape:

    - `(64, 128)`

    This means the model stores one learned vector of width 128 for each
    possible position from 0 to 63.

    If the current batch has token ids of shape:

    - `(16, 40)`

    then the current sequence length is 40, so this module selects the
    positional embeddings for positions 0 to 39 and returns them in
    broadcast-ready form with shape:

    - `(1, 40, 128)`

    These can then be added to token embeddings of shape:

    - `(16, 40, 128)`
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialise the positional-embedding module.

        Parameters
        ----------
        config : ModelConfig
            Model configuration describing the positional embedding table.

        Notes
        -----
        - We validate the configuration immediately so that invalid settings fail
          early and clearly.

        - In particular, positional embeddings require:

            - a positive context length
            - a positive embedding dimension
        """
        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError("config must be an instance of ModelConfig.")

        config.validate()

        self.context_length = config.context_length
        self.embedding_dim = config.embedding_dim

        # Create the learnable position-embedding table.
        #
        # Shape:
        #   (context_length, embedding_dim)
        #
        # Each row corresponds to one absolute position in the sequence:
        #
        # - row 0  -> position 0
        # - row 1  -> position 1
        # - row 2  -> position 2
        # - ...
        # - row 63 -> position 63   (if context_length = 64)
        self.embedding = nn.Embedding(
            num_embeddings=self.context_length,
            embedding_dim=self.embedding_dim,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Build positional embeddings for the current batch sequence length.

        Parameters
        ----------
        token_ids : Tensor
            Batch of integer token ids with shape `(B, T)`.

            The actual token values are not used here.
            We use this tensor only to determine:

            - which device to place the position indices on
            - how long the current sequence is

        Returns
        -------
        Tensor
            Positional embeddings with shape `(1, T, C)` where:

            - `1` is a broadcastable batch dimension
            - `T` is sequence length
            - `C` is embedding dimension

        Notes
        -----
        The logic is:

        1. inspect the input shape `(B, T)`
        2. create position indices `[0, 1, ..., T - 1]`
        3. look up the learned vector for each position
        4. add a leading batch dimension so the result can be broadcast later

        Example
        -------
        Suppose:

        - `token_ids.shape = (16, 64)`
        - `embedding_dim = 128`

        Then:

        - positions has shape `(64,)`
        - raw position embeddings have shape `(64, 128)`
        - returned positional embeddings have shape `(1, 64, 128)`

        In plain language:

        - every position in the sequence gets its own learned vector
        - the same position vectors are shared across all items in the batch
        """
        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor.")
        
        # Token ids should arrive as a batch of sequences with shape `(B, T)`
        #   - If a tensor with the wrong rank reaches this module, it usually means
        #     something earlier in the data or model pipeline is misaligned.
        assert_rank(token_ids, expected_rank=2, tensor_name="token_ids")

        if token_ids.dtype != torch.long:
            raise TypeError(
                "token_ids must have dtype torch.long, "
                f"but received dtype {token_ids.dtype}."
            )
        
        batch_size, sequence_length = token_ids.shape

        # The batch size is not directly used when constructing the positional
        # embeddings, because positions are shared across every sequence in the
        # batch
        #   - It is unpacked here because it makes the input shape easier to
        #     read and explain
        _ = batch_size

        # The learned table only contains positions from:
        #
        #   0 up to context_length - 1
        #
        # So the current sequence length must not exceed the configured maximum
        # context length
        if sequence_length > self.context_length:
            raise ValueError(
                "sequence length exceeds the configured context length. "
                f"Received sequence length {sequence_length}, but the maximum "
                f"supported context length is {self.context_length}."
            )
        
        # Create explicit integer position indices:
        #
        #   [0, 1, 2, ..., sequence_length -1]
        #
        # Shape:
        #
        #   (T,)
        #
        # We place these indices on the same device as the input token ids so
        # that CPU/GPU mismatches do not occur later
        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
            dtype=torch.long,
        )

        # Look up the learned positional vector for each position index
        #
        # Input shape:
        #
        #   positions -> (T,)
        #
        # Output shape:
        #
        #   position_embeddings -> (T, C)
        position_embeddings = self.embedding(positions)

        # Add a leading singleton batch dimension
        #
        # Before unsqueeze:
        #
        #   (T, C)
        #
        # After unsqueeze:
        #
        #   (1, T, C)
        #
        # This is useful because later we will add these positional embeddings
        # to token embeddings of shape `(B, T, C)`. PyTorch broadcasting will
        # automatically copy the same positional information across all `B`
        # items in the batch.
        #
        # For example, if `batch_size = 16` `sequence_length = 64` and
        # `embedding_dim = 128`:
        #
        #       token_embeddings.shape == (16, 64, 128)
        #       position_embeddings.shape == (1, 64, 128)
        position_embeddings = position_embeddings.unsqueeze(0)

        # This check is mostly educational and defensive. 
        #   - `nn.Embedding` is reliable, but making the intended output shape 
        #     explicit produces clearer debug messages if something upstream changes.
        if position_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                "Positional embedding output has the wrong final dimension. "
                f"Expected {self.embedding_dim}, received shape "
                f"{format_shape(position_embeddings)}."
            )

        return position_embeddings