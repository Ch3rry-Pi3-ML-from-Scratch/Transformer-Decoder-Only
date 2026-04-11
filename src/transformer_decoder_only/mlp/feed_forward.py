"""
Feed-forward network module for the educational decoder-only transformer project.

This module is intentionally small, but it provides the second major sublayer
inside a transformer decoder block. It gives the rest of the repository a stable
way to talk about:

- position-wise non-linear transformation
- expanding and contracting the embedding dimension
- adding modelling capacity after attention
- keeping the output shape compatible with residual connections

Keeping feed-forward logic in one place makes the project easier to extend because:

- `attention` can focus on mixing information across token positions
- `blocks.decoder_block` can focus on composing attention, MLP, residuals, and normalisation
- `models.decoder_transformer` can focus on stacking complete decoder blocks
- `tests` can verify shape preservation separately from the full model

In plain language:

- attention lets token positions communicate with each other
- the feed-forward network processes each token position independently
- it gives the model extra non-linear capacity after attention has mixed context

Notes
-----
- A standard transformer feed-forward network is often a two-layer MLP:

    Linear(C -> 4C)
    GELU
    Linear(4C -> C)
    Dropout

- The expansion from `C` to `4C` gives the model a wider intermediate space in
  which to transform each position.
- The contraction from `4C` back to `C` makes the output shape match the input
  shape again.
- Matching the input and output shape is important because the decoder block
  will later use a residual connection:

    x = x + feed_forward(...)

- This module does not mix information across sequence positions.
  If the input has shape:

    (B, T, C)

  then the same MLP is applied independently at each of the `T` positions.

Shape notation
--------------
Let:

    `B` = batch size
    `T` = sequence length
    `C` = embedding dimension

Then:

- input hidden states:      (B, T, C)
- expanded hidden states:   (B, T, 4C)
- output hidden states:     (B, T, C)

Example
-------
If:

- `embedding_dim = 128`

then the feed-forward network maps:

    (B, T, 128) -> (B, T, 512) -> (B, T, 128)

The batch size and sequence length are preserved throughout.
"""

from torch import Tensor
from torch import nn

from transformer_decoder_only.config.default import ModelConfig
from transformer_decoder_only.utils.shapes import assert_last_dim, assert_rank, format_shape

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used inside a decoder block.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing at least:

        - `embedding_dim`
        - `dropout`
        - `use_bias`

    Attributes
    ----------
    embedding_dim : int
        Full model embedding dimension, usually called `C`.
    hidden_dim : int
        Intermediate MLP width, usually set to `4 * C`.
    network : nn.Sequential
        The feed-forwards network itself.

    Notes
    -----
    - The conceptual job of this module is to transform each token representation
      after attention has gathered contextual information.
    - In plain language:

        - attention decides what information each position should read from the context
        - the feed-forwards network transforms the resulting representation at each position

    - Importantly, this module preserves shape:

        (B, T, C) -> (B, T, C)

    - That makes it compatible with residual addition in the decoder block.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialise the feed-forward network.

        Parameters
        ----------
        config : ModelConfig
            Model configuration describing the hidden width and dropout rate.

        Notes
        -----
        - The intermediate width is set to `4 * embedding_dim`, following the
          common transformer design pattern.
        - For example, if:

            embedding_dim = 128
          
          then:

            hidden_dim = 512
        """

        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError("config must be an instance of ModelConfig.")
        
        config.validate()

        self.embedding_dim = config.embedding_dim
        self.hidden_dim = 4 * config.embedding_dim

        # The feed-forward network is applied independently at each sequence position
        #
        # Shape flow:
        #
        #   (B, T, C)   ->  Linear  ->  (B, T, 4C)
        #   (B, T, 4C)  ->  GELU    ->  (B, T, DC)
        #   (B, T, 4C)  ->  Linear  ->  (B, T, C)
        #   (B, T, C)   ->  Dropout ->  (B, T, C)
        self.network = nn.Sequential(
            nn.Linear(
                in_features=self.embedding_dim,
                out_features=self.hidden_dim,
                bias=config.use_bias,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.embedding_dim,
                bias=config.use_bias,
            ),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Apply the position-wise feed-forward network.

        Parameters
        ----------
        hidden_states : Tensor
            Input hidden states with shape `(B, T, C)` where:

            - `B` is batch size
            - `T` is sequence length
            - `C` is embedding dimension

        Returns
        -------
        Tensor
            Transformed hidden states with shape `(B, T, C)`.

        Notes
        -----
        - Even though the tensor has a sequence dimension, this MLP does not 
          perform attention or communication between positions.
        - PyTorch's linear layers operate on the final dimension, so the same
          learned transformation is applied independently to every `(B, T)` token
          position.

        Example
        -------
        If input hidden states have shape:

            (16, 64, 128)

        then the internal transformation is:

            (16, 64, 128) -> (16, 64, 512) -> (16, 64, 128)

        In plain language:

        - every token position starts with a 128-dimensional vector
        - it is expanded to 512 dimensions
        - a non-linearity is applied
        - it is projected back to 128 dimensions
        """

        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor.")
        
        # Feed-forward expects hidden states, not raw token ids
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

        output = self.network(hidden_states)

        if output.shape != hidden_states.shape:
            raise ValueError(
                "FeedForward output must have the same shape as its input "
                "so it can be used in a residual connection. "
                f"Input shape: {format_shape(hidden_states)}, "
                f"output shape: {format_shape(output)}."
            )

        return output
