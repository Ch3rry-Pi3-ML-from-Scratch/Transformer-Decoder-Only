"""
Multi-head attention module for the educational decoder-only transformer project.

This module builds directly on the single self-attention head. It gives the rest
of the repository a stable way to talk about:

- running several attention heads over the same hidden states
- concatenating per-head outputs
- projecting the combined result back into the model embedding space
- keeping the attention block shape-compatible with residual connections

Keeping multi-head attention in its own module makes the project easier to extend because:

- `attention.self_attention_head` can focus on one head's Q, K, V mechanics
- `blocks.decoder_block` can focus on residual connections and layer normalisation
- `models.decoder_transformer` can focus on stacking full decoder blocks
- `tests` can verify multi-head output shape separately from the full model

In plain language:

- this module answers the question,

    "How do we use several attention heads at once?"

- each head gets to learn a different way of looking at the same sequence
- their outputs are then joined together and mixed with a final projection

Notes
-----
- This implementation is intentionally pedagogical rather than maximally efficient.
- In many production transformer implementations, all heads are computed using
  larger combined Q, K, and V projection matrices, then reshaped internally.
  
  Here, we use a list of separate `SelfAttentionHead` modules because it makes
  the idea easier to inspect:

    - head 0 attents in one learned way
    - head 1 attends in another learned way
    - head 2 attends in another learned way
    - and so on

- If:

    embedding_dim = 128
    num_heads = 4

  then each head has:

    head_dim = 128 // 4 = 32

- Each head returns a tensor of shape:

    (B, T, 32)

- Concatenating 4 heads along the final dimension gives:

    (B, T, 128)

- The final output projection then maps:

    (B, T, 128) -> (B, T, 128)

  so the module can fit neatly inside a residual connection.

- One common beginner mistake is to think that each head reduces the sequence.
  It does not.

  Each head keeps the same batch and sequence dimensions:

    (B, T, C) -> (B, T, D)

  So the sequence length `T` is preserved. What changes is the final feature
  dimension.

- Another common beginner mistake is to think the heads are added together.
  They are not added. They are concatenated along the final feature dimension.

  For example, with 4 heads of width 32:

      [(B, T, 32), (B, T, 32), (B, T, 32), (B, T, 32)]

  becomes:

      (B, T, 128)

  Addition happens later in the decoder block when we use a residual connection:

      x = x + attention_output
"""

import torch
from torch import Tensor
from torch import nn

from transformer_decoder_only.attention.self_attention_head import SelfAttentionHead
from transformer_decoder_only.config.default import ModelConfig
from transformer_decoder_only.utils.shapes import assert_last_dim, assert_rank, format_shape

class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention module.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing at least:

        - `embedding_dim`
        - `num_heads`
        - `dropout`
        - `use_bias`

    Attributes
    ----------
    embedding_dim : int
        Full model embedding dimension, usually called `C`.
    num_heads : int
        Number of parallel attention heads.
    head_dim : int
        Per-head feature dimension, usually called `D`.
    heads : nn.ModuleList
        Collection of individual `SelfAttentionHead` modules.
    output_projection : nn.Linear
        Linear layer that mixes the concatenated head outputs
    output_dropout : nn.Dropout
        Dropout applied after the output projection.

    Notes
    -----
    - The conceptual job of multi-head attention is:

        1. run several attention heads over the same input
        2. concatenate their outputs
        3. mix the concatenated representation with a final linear projection

    - In plain language:

        - every head sees the same input sequence
        - every head starts from the full hidden representation
        - every head learns its own lower-dimensional view of that representation
        - the head outputs are placed side by side through concatenation
        - the output projection mixes those side-by-side views

    - A key detail is that a head does not shorted the sequence.
      It changes the feature width only.

    - For one head:

        (B, T, T) -> (B, T, D)

    - Here

        - `B` is unchanged
        - `T` is unchanged
        - `C` becomes `D`

    Tensor shapes
    -------------
    Let:

        `B` = batch size
        `T` = sequence length
        `C` = embedding dimension
        `H` = number of heads
        `D` = head dimension

    Then:

    - input hidden states:                      (B, T, C)
    - each head output:                         (B, T, D)
    - concatenated head output:                 (B, T, H * D)
        - because `H * D = C`, this becomes:    (B, T, C)
    - final projected output:                   (B, T, C)

    Example
    -------
    Suppose:

        embedding_dim = 8
        num_heads = 2

    then each head uses

        head_dim = embedding_dim / num_heads = 8 / 2 = 4

    Now suppose the input to multi-head attention has shape:

        (B, T, C) = (1, 3, 8)

    this means:

    - batch_size = 1
    - sequence_size = 3
    - each token is represented by a vector of width 8

    A tiny input example could look like this:

        x =
        [
            [
                [1.0, 0.5, 0.2, 0.1, 0.0, 0.3, 0.8, 0.4],   # position 0
                [0.9, 0.1, 0.7, 0.2, 0.3, 0.6, 0.0, 0.5],   # position 1
                [0.4, 0.8, 0.2, 0.9, 0.1, 0.0, 0.3, 0.7],   # position 2
            ]
        ]

    Each attention head does not simply take half of these 8 numbers directly.
    Instead, each head applies its own learned linear projections to map the
    8-dimensional token vectors into its own 4-dimensional query, key, and value
    spaces.

    So each head receives the same input shape:

        (1, 3, 8)

    but produces an output of shape:

        (1, 3, 4)

    For two heads, we therefore get:

        head_1_output.shape = (1, 3, 4)
        head_2_output.shape = (1, 3, 4)

    Conceptually, the outputs might look like this:

    head_1_output =
    [
        [
            [0.2, 0.7, 0.1, 0.5],
            [0.3, 0.6, 0.2, 0.4],
            [0.5, 0.4, 0.3, 0.6],
        ]
    ]

    head_2_output =
    [
        [
            [0.8, 0.1, 0.9, 0.2],
            [0.7, 0.3, 0.6, 0.1],
            [0.4, 0.5, 0.2, 0.8],
        ]
    ]

    Concatenating these along the final dimension gives:

        concatenated.shape = (1, 3, 8)

    because the last dimension is rebuilt as:

        4 + 4 = 8

    So the concatenated output has the same overall feature width as the original
    input.

    The final output projection then maps:

    - `(1, 3, 8)` -> `(1, 3, 8)`

    This is important because the decoder block later uses a residual connection:

        `x = x + attention_output`

    and that addition requires both tensors to have exactly the same shape.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialise the multi-head attention module.

        Parameters
        ----------
        config : ModelConfig
            Model configuration describing the attention dimensions.

        Notes
        -----
        - We build `num_heads` separate self-attention heads.
        - This is not the most compact or fastest implementation, but it is very
          readable and makes the architecture easier to understand.
        """

        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError("config must be an instance of ModelConfig.")
        
        config.validate()

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # Build the individual attention heads
        #
        # Each head receives the same input hidden states of shape `(B, T, C)`
        #
        # Important:
        #   The head does not receive a fixed slice of the input embedding.
        #   Instead, each head learns its own Q, K, and V projections from the
        #   full embedding dimension `C` down to the smaller head dimension `D`
        # Each head returns its own output of shape `(B, T, D)`
        self.heads = nn.ModuleList(
            SelfAttentionHead(config) for _ in range(self.num_heads)
        )

        # After concatenating all heads, the feature dimension is:
        #
        #   num_heads * head_dim
        #
        # With our default values:
        #
        #   4 * 32 = 128
        #
        # Because `head_dim = embedding_dim // num_heads`, this equals:
        #
        #   embedding_dim
        #
        # The output projection then mixes information across heads while
        # preserving the model dimension.
        self.output_projection = nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.embedding_dim,
            bias=config.use_bias,
        )

        # Dropout after the output projection is common in transformer blocks
        #   - It does not change the tensor shape.
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Apply causal multi-head self-attention.

        Parameters
        ----------
        hidden_states : Tensor
            Input states with shape `(B, T, C)` where:

            - `B` is batch size
            - `T` is sequence length
            - `C` is embedding dimensions

        Returns
        -------
        Tensor
            Multi-head attention output with shape `(B, T, C)`

        Notes
        -----
        - The forward pass follows these steps:

            1. pass the same hidden states into each attention head
            2. collect each head output of shape `(B, T, D)`
            3. concatenate head outputs along the feature dimension
            4. project the concatenated tensor back into embedding space
            5. apply output dropout

        - In plain language:

            - each head looks at the sequence in its own learned way
            - concatenation puts those separate views side by side
            - the output projection mixes those views back together
        """
        
        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor.")
        
        # Step 1: Multi-head attention expects hidden states, not raw token ids
        #
        # Expected shape:
        #
        #   (B, T, C)
        assert_rank(hidden_states, expected_rank=3, tensor_name="hidden_states")
        assert_last_dim(
            hidden_states,
            expected_last_dim=self.embedding_dim,
            tensor_name="hidden_states",
        )

        # Step 2: Run each head independently over the same hidden states
        #
        # Each output has shape:
        #
        #   (B, T, D)
        #
        # With default values:
        #
        # (B, T, 32)
        head_outputs = [head(hidden_states) for head in self.heads]

        # Step 3: Concatenate the head outputs along the final feature dimension
        #
        # If we have 4 heads, and each head has dimension 32:
        #
        #   [(B, T, 32), (B, T, 32), (B, T, 32), (B, T, 32)]
        #
        # becomes:
        #
        #   (B, T, 128)
        #
        # This is concatenation, not addition.
        concatenated = torch.cat(head_outputs, dim=-1)

        if concatenated.shape[-1] != self.embedding_dim:
            raise ValueError(
                "Concatenated attention heads have the wrong final dimension. "
                f"Expected {self.embedding_dim}, received shape {format_shape(concatenated)}."
            )
        
        # Mix information across heads with a learned output projection.
        #
        # Shape:
        #   (B, T, C) -> (B, T, C)
        projected = self.output_projection(concatenated)

        # Apply dropout after projection.
        #
        # Shape:
        #   (B, T, C) -> (B, T, C)
        output = self.output_dropout(projected)

        if output.shape != hidden_states.shape:
            raise ValueError(
                "MultiHeadAttention output must have the same shape as its input "
                "so it can be used in a residual connection. "
                f"Input shape: {format_shape(hidden_states)}, "
                f"output shape: {format_shape(output)}."
            )

        return output


