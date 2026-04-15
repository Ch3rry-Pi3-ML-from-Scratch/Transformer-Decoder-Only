"""
Decoder block module for the education decoder-only transformer project.

This module is where the transformer starts to come together as a recognisable
architecture. It gives the rest of the repository a stable way to talk about:

- layer normalisation
- causal multi-head self-attention
- feed-forward transformation
- residual connections
- stacking reusable decoder blocks

Keeping the decoder block in its own module makes the project easier to extend because:

- `attention` can focus on attention mechanics
- `mlp` can focus on feed-forward transformations
- `models.decoder_transformer` can focus on stacking blocks into a full model
- `tests` can verify block-level shape preservation separately from full training

In plain language:

- this module answers the question,

    "How do attention and feed-forward layers fit together?"

- it combines the two main transformer sublayers into one reusable block.

Notes
-----
- A decoder-only transformer is usually built by stacking several decoder blocks.
- Each block contains two main sublayers:

    1. causal multi-head self-attention
    2. position-wise feed-forward network

- Each sublayer is wrapped with:

    - layer normalisation
    - a residual connection

- This implementation uses the common "pre-norm" structure:

    x = x + attention(layer_norm_1(x))
    x = x + feed_forward(layer_norm_2(x))

- "Pre-norm" means the normalisation happens before each sublayer.
- Residual connections are important because they let the original hidden state
  flow around a sublayer and be added back afterwards.
- In plain language:

    output = original_information + new_information

- Shape preservation is essential.
  If `x` has shape:

    (B, T, C)

  then each sublayer must also return:

    (B, T, C)

  so the residual addition is valid.
"""

from torch import Tensor
from torch import nn

from transformer_decoder_only.attention.multi_head_attention import MultiHeadAttention
from transformer_decoder_only.config.default import ModelConfig
from transformer_decoder_only.mlp.feed_forward import FeedForward
from transformer_decoder_only.utils.shapes import assert_last_dim, assert_rank, format_shape

class DecoderBlock(nn.Module):
    """
    One pre-norm decoder block for a decoder-only transformer.

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
    attention_norm : nn.LayerNorm
        Layer normalisation applied before multi-head attention.
    attention : MultiHeadAttention
        Causal multi-head self-attention sublayer.
    feed_forward_norm : nn.LayerNorm
        Layer normalisation applied before the feed-forward network.
    feed_forward : FeedForward
        Position-wise feed-forward sublayer.

    Notes
    -----
    - The block follows this pattern:

        x = x + attention(attention_norm(x))
        x = x + feed_forward_norm(x))

    - This means:

        - normalise before attention
        - apply attention
        - add the attention result back to the original stream
        - normalise before feed-forward
        - apply feed-forward
        - add the feed-forward result back to the stream

    Tensor shapes
    -------------
    Let:

    - `B` = batch size
    - `T` = sequence length
    - `C` = embedding dimension

    Then the block preserves shape:

    - input: `(B, T, C)`
    - output: `(B, T, C)`

    Example
    -------
    If the input hidden states have shape:

        `(16, 64, 128)`

    then the decoder block also returns:

        `(16, 64, 128)`

    This is what allows many decoder blocks to be stacked one after another.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialise the doceoder block.

        Parameters
        ----------
        config : ModelConfig
            Model configuration describing the block dimensions.

        Notes
        -----
        - We create two independent layer-normalisation modules because attention
          and feed-forward are separate sublayers with separate inputs.
        - Both normalise over the final feature dimension `C`.
        """

        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError("config must be an instance of ModelConfig.")
        
        config.validate()

        self.embedding_dim = config.embedding_dim

        # LayerNorm normalises across the final feature dimension
        #
        # Input shape:
        #
        #   (B, T, C)
        #
        # Output shape:
        #
        #   (B, T, C)
        self.attention_norm = nn.LayerNorm(
            normalized_shape=self.embedding_dim,
            elementwise_affine=True,
            bias=config.use_bias,
        )

        # Causal multi-head attention lets each token position read from the
        # current and previous positions, but not future positions
        #
        # Shape:
        #
        #   (B, T, C) -> (B, T, C)
        self.attention = MultiHeadAttention(config)

        # A second LayerNorm is used before the feed-forward sublayer
        #   - This follows the pre-norm transformer block pattern:
        #
        #   x = x + attention(norm_1(x))
        #   x = x + feed_forward(norm_2(x))
        self.feed_forward_norm = nn.LayerNorm(
            normalized_shape=self.embedding_dim,
            elementwise_affine=True,
            bias=config.use_bias,
        )

        # Position-wise MLP applied after attention
        #
        # Shape:
        #
        #   (B, T, C) -> (B, T, C)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Apply one decoder block.

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
            Output hidden states with shape `(B, T, C)`.

        Notes
        -----
        - The forward pass has two residual updates:

            1. attention residual update
            2. feed-forward residual update

        - Written compactly:

            x = x + attention(norm_1(x))
            x = x + feed_forward(norm_2(x))

        - In plain language:

            - keep the original stream
            - compute a new attention-based update
            - add that update to the stream
            - compute a new feed-forward update
            - add that update to the stream
        """

        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor.")
        
        # The decoder block expects hidden states, not raw token ids
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

        # First residual path: causal self-attention
        attention_input = self.attention_norm(hidden_states)
        attention_output = self.attention(attention_input)
        hidden_states = hidden_states + attention_output

        # Second residual path: position-wise feed-forward network
        feed_forward_input = self.feed_forward_norm(hidden_states)
        feed_forward_output = self.feed_forward(feed_forward_input)
        hidden_states = hidden_states + feed_forward_output

        if hidden_states.shape[-1] != self.embedding_dim:
            raise ValueError(
                "DecoderBlock output has the wrong final dimension. "
                f"Expected {self.embedding_dim}, received shape {format_shape(hidden_states)}."
            )

        return hidden_states