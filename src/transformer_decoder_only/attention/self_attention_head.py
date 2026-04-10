"""
Single self-attention head for the educational decoder-only transformer project.

This module is intentionally explicit because it contains the central mechanism 
of the transformer. It gives the rest of the repository a stable way to talk
about:

- query projections
- key projections
- value projections
- scaled dot-product attention
- causal masking
- attention weights
- weighted sums of value vectors

Keeping one attention head in its own module makes the project easier to extend because:

- `attention.multi_head_attention` can focus on combining several heads
- `blocks.decoder_block` can focus on residual connections and layer normalisation
- `models.decoder_transformer` can focus on composing the full model
- `tests` can verify attention behaviour separately from the rest of the network

In plain language:

- this module answers the question, 

    "For each token position, which earlier positions should it pay attention to?"

- it implements one attention head manually so the mechanism is visible.

Notes
-----
- This module does not use `torch.nn.MultiheadAttention`.
- This module does not use `torch.nn.Transformer`.
-Instead, we manually implement the core attention steps:

    1. Project hidden states into queries `Q`.
    2. Project hidden states into keys `K`.
    3. Project hidden states into values `V`.
    4. Compute dot-product attention scores with `Q @ K.transpose(-2, -1)`.
    5. Scale scores by `sqrt(head_dim)`
    6. Apply a causal mask so positions cannot attend to future positions.
    7. Apply softmax to get attention weights.
    8. Use attention weights to compute a weighted sum of the value vectors.

Shape notation
--------------

Let:

    `B` = batch size
    `T` = sequence length
    `C` = full embedding dimension
    `D` = head dimension

Then the key shapes are:

- input hidden states:      (B, T, C)
- queries `Q`:              (B, T, D)
- keys `K`:                 (B, T, D)
- values `V`:               (B, T, D)
- attention scores:         (B, T, T)
- attention weights:        (B, T, T)
- head output:              (B, T, D)

The attention score matrix has shape `(B, T, T)` because every query position
scores every key position.
"""

import math

import torch
from torch import Tensor
from torch import nn

from transformer_decoder_only.attention.causal_mask import (
    apply_causal_mask,
    build_causal_mask,
)
from transformer_decoder_only.config.default import ModelConfig
from transformer_decoder_only.utils.shapes import assert_last_dim, assert_rank, format_shape

class SelfAttention(nn.Module):
    """
    One causal self-attention head.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing at least:

        - `context_length`
        - `embedding_dim`
        - `num_heads`
        - `dropout`
        - `use_bias`
    
    Attributes
    ----------
    embedding_dim : int
        Full model embedding dimension, usually called `C`.
    head_dim : int
        Feature dimension used by this attention head, usually called `D`.
    context_length : int
        Maximum supported sequence length.
    query_projection : nn.Linear
        Linear layer that maps hidden states to query vectors.
    key_projection : nn.Linear
        Linear layer that maps hidden states to key vectors.
    value_projection : nn.Linear
        Linear layer that maps hidden states to value vectors.
    attention_dropout : nn.Dropout
        Dropout applied to attention weights during training.

    Notes
    -----
    - Self-attention means that the sequence attends to itself.
    - In plain language:

        - each position creates a query: "What am I looking for?"
        - each position creates a key: "What information do I contain?"
        - each position creates a value: "What information should I pass along?"
        - queries are compared with keys to decide attention strength
        - attention weights are used to combine values

    - Because this is a decoder-only language model, the attention must be causal:

        - position 0 can only attend to position 0
        - position 1 can attend to positions 0 and 1
        - position 2 can attend to positions 0, 1, and 2
        - and so on

    - This prevents the model from cheating by looking at future tokens.

    Example
    -------
    Suppose the current input sequence has length 3, so the model is processing
    three positions:

        - position 0
        - position 1
        - position 2

    After earlier embedding steps, assume this attention head receives an input
    tensor `x` of shape:

        (B, T, C)

    For example, if:

        B = 2   (batch size)
        T = 3   (sequence length)
        C = 12  (full embedding dimension)

    then:

        x.shape = (2, 3, 12)

    If the model uses:

        num_heads = 3

    then each head works on:

        head_dim = C / num_heads = 12 / 4 = 3

    So this single attention head project the input into:

        - queries of shape  `(2, 3, 4)`
        - keys of shape     `(2, 3, 4)`
        - values of shape   `(2, 3, 4)`

    For each position, the query vector is compared with the key vectors from the
    allowed positions in the sequence.

    Because this is causal attention:

        - position 0 compares only with position 0
        - position 1 compares with positions 0 and 1
        - position 2 compares with positions 0, 1, and 2

    These comparisons produce attention scores, which are converted into attention
    weights using softmax. The weights are then used to form a weighted combination
    of the value vectors.

    So, for position 2, the head is conceptually doing something like:

    - compare `q_2` with `k_0`, `k_1`, and `k_2`
    - turn those scores into weights
    - use those weights to combine `v_0`, `v_1`, and `v_2`

    The result is a new vector for position 2 that mixes information from the
    visible earlier positions.

    Numeric Example
    ---------------
    Suppose that, for one sequence in the batch, this head (remember head has dimension
    4) produces the following vectors for position 2 and the visible earlier positions:

        q_2 = [1.0, 1.0, 0.0, 0.0]

        k_0 = [1.0, 0.0, 0.0, 0.0]
        k_1 = [0.0, 1.0, 0.0, 0.0]
        k_2 = [1.0, 1.0, 0.0, 0.0]

        v_0 = [2.0, 0.0, 0.0, 0.0]
        v_1 = [0.0, 3.0, 0.0, 0.0]
        v_2 = [1.0, 1.0, 0.0, 0.0]

    The raw dot-product attention scores for position 2 are:

        q_2 · k_0 = 1.0
        q_2 · k_1 = 1.0
        q_2 · k_2 = 2.0

    So the unscaled score vector is:

        [1.0, 1.0, 2.0]

    Because this is scaled dot-product attention, we divide by:

        sqrt(head_dim) = sqrt(4) = 2

    which gives scaled scores:

        [0.5, 0.5, 1.0]

    Applying softmax gives attention weights approximately:

        [0.27, 0.27, 0.46]

    This means position 2 attends:

        - about 27% to position 0
        - about 27% to position 1
        - about 46% to position 2

    The output for position 2 is then the weighted sum of the value vectors:

        0.27 * v_0 + 0.27 * v_1 + 0.46 * v_2

    which is approximately:

          0.27 * [2.0, 0.0, 0.0, 0.0]
        + 0.27 * [0.0, 3.0, 0.0, 0.0]
        + 0.46 * [1.0, 1.0, 0.0, 0.0]

        = [1.00, 1.27, 0.0, 0.0]

    So the new representation for position 2 becomes a mixture of information from
    positions 0, 1, and 2, with the strongest contribution coming from position 2
    itself.

    The result is a new vector for position 2 that mixes information from the
    visible earlier positions.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialise one self-attention head.

        Parameters
        ----------
        config : ModelConfig
            Model configuration describing the attention dimensions.

        Notes
        -----
        - The full embedding dimension is split across multiple heads.
        - If:

            embedding_dim = 128
            num_heads = 4

          then:

            head_dim = 32

        - This class represents only one of those heads.
        """

        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError("config must be an instance of ModelConfig.")
        
        config.validate()

        self.embedding_dim = config.embedding_dim
        self.head_dim = config.head_dim
        self.context_length = config.context_length

        # The query projection answers:
        #
        #   "What is this position looking for?"
        #
        # Shape transformation:
        #
        #   (B, T, C) -> (B, T, D)
        self.query_projection = nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.head_dim,
            bias=config.use_bias,
        )

        # The key projection answers:
        #
        #   "What does this position contain that another position might match?"
        #
        # Shape transformation:
        #
        #   (B, T, C) -> (B, T, D)
        self.key_projection = nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.head_dim,
            bias=config.use_bias,
        )

        # The value projection answers:
        #
        #   "What information should this position contribute if attended to?"
        #
        # Shape transformation:
        #   (B, T, C) -> (B, T, D)
        self.value_projection = nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.head_dim,
            bias=config.use_bias,
        )

        # Dropout is applied to the attention weights during training
        #   - This means that some attention connections are randomly weakened or
        #     removed, which can help reduce overfitting in larger settings.
        self.attention_dropout = nn.Dropout(config.dropout)


    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Apply one causal self-attention head.

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
            Output of this attention head with shape `(B, T, D)` where:

            - `D` is the per-head dimension

        Notes
        -----
        - The attention calculation follows these steps:

            1. Project the input into `Q`, `K`, and `V`.
            2. Compute attention scores using dot products between `Q` and `K`.
            3. Scale the scores by `sqrt(D)`.
            4. Apply the causal mask.
            5. Apply softmax to obtain attention weights.
            6. Multiply the attention weights by `V`.

        - In plain language:
            
            - scores decide where each position wants to look
            - the mask blocks future positions
            - softmax turns scores into probabilities
            - the probabilities blend the value vectors 
        """

        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor.")
        
        # Hidden states should arrive after token and positional embeddings have
        # already combined.
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

        batch_size, sequence_length, embedding_dim = hidden_states.shape

        # We unpack these names explicitly because the shapes are central
        # to understanding attention
        _ = batch_size
        _ = embedding_dim

        if sequence_length > self.context_length:
            raise ValueError(
                "sequence length exceeds the configured context length. "
                f"Received sequence length {sequence_length}, but the maximum "
                f"supported context length is {self.context_length}."
            )
        
        # Step 1: project hidden states into Q, K, and V
        #
        # Input shape:
        #
        #   hidden_states -> (B, T, C)
        #
        # Output shapes:
        #
        #   queries -> (B, T, D)
        #   keys    -> (B, T, D)
        #   values  -> (B, T, D)
        queries = self.query_projection(hidden_states)
        keys = self.key_projection(hidden_states)
        values = self.value_projection(hidden_states)

        # Step 2: compute raw dot-product attention scores
        #
        # We compare each query vector with each key vector
        #
        # keys.transpose(-2, -1) changes:
        #   keys from (B, T, D)
        #        to   (B, D, T)  
        #
        # For example, if: 
        # 
        #   keys.shape == (B, T, D) = (1, 3, 2)
        #
        # Then:
        #
        #   keys.transpose(-2, -1).shape == (B, D, T) = (1, 2, 3)
        #
        # In matrix form:
        # 
        #    keys = [
        #        [
        #            [1, 2],   # position 0
        #            [3, 4],   # position 1
        #            [5, 6],   # position 2
        #        ]
        #    ]
        #
        # then:
        #
        #    keys.transpose(-2,-1) = [
        #        [
        #            [1, 3, 5],
        #            [2, 4, 6],
        #        ]
        #    ]
        #
        # After this, we then take the dot product:
        #
        #   queries @ keys.transpose(-2, -1)
        #
        # giving:
        #
        #   (B, T, D) @ (B, D, T) -> (B, T, T)
        # 
        # Each row in the final `(T, T)` matrix says:
        # 
        #   "For this query position, how much does each key position match?"
        # 
        attention_scores = queries @ keys.transpose(-2, -1)

        # Step 3: scale the attention scores
        # 
        # Dot products tend to grow in magnitude when the vector dimension `D`
        # grows. Large scores can make softmax too sharp early in the training.
        # 
        # Dividing by sqrt(D) keeps the scale of the scores more stable.
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Step 4: build and apply the causal mask
        # 
        # The mask shape is:
        #   
        #   (1, T, T)
        # 
        # The attention score shape is:
        # 
        #   (B, T, T)
        # 
        # PyTorch broadcasts the leading `1` across the batch dimension.
        causal_mask = build_causal_mask(
            sequence_length=sequence_length,
            device=hidden_states.device,
        ) 
        attention_scores = apply_causal_mask(attention_scores, causal_mask)

        # Step 5: convert masked scores into attention weights
        #
        # We apply softmax over the final dimension because each query position
        # must distribute its attention over key positions.
        #
        # Shape:
        #
        #   attention_weights -> (B, T, T)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights during training
        #
        # This does not change the shape:
        #
        #   (B, T, T) -> (B, T, T)
        attention_weights = self.attention_dropout(attention_weights)

        # Step 6: use the attention weights to blend value vectors.
        #
        # Shape:
        #
        #   attention_weights -> (B, T, T)
        #   values            -> (B, T, D)
        #
        # Matrix multiply:
        #
        #   (B, T, T) @ (B, T, D) -> (B, T, D)
        #
        # In plain language:
        #
        #   each position now receives a weighted mixture of value vectors from
        #   the positions it was allowed to attend to.
        head_output = attention_weights @ values

        if head_output.shape != (hidden_states.shape[0], sequence_length, self.head_dim):
            raise ValueError(
                "SelfAttentionHead produced an unexpected output shape. "
                f"Expected {(hidden_states.shape[0], sequence_length, self.head_dim)}, "
                f"received {format_shape(head_output)}."
            )

        return head_output
        