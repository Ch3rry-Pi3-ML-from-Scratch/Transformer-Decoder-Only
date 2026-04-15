"""
Tests for causal masking and attention modules in the educational transformer project.

This module is intentionally focused on the attention stack. It helps us verify
the core transformer mechanism before we build the full decoder-only model.

That matters because attention is one of the easiest places to make subtle
shape or masking mistakes.

If attention is wrong, then:

- the model may accidentally look at future tokens
- tensor shapes may stop matching residual connections
- multi-head concatenation may rebuild the wrong feature dimension
- later model-level tests may fail in ways that are harder to interpret

So these tests are designed to check the attention pieces directly.

In plain language:

- does the causal mask have the expected lower-triangle pattern?
- does masking replace future-position scores with `-int`?
- does one self-attention head return the expected shape?
- does multi-head attention return original embedding shape?
- do invalid shapes fail clearly?

Notes
-----
- These tests use small hand-checkable tensors where possible.
- The goal is not to test PyTorch itself.
- The goal is to verify that our attention modules use PyTorch correctly and
  preserve the transformer shape conventions we expect.
"""

import pytest
import torch

from transformer_decoder_only.attention.causal_mask import (
    apply_causal_mask,
    build_causal_mask,
)
from transformer_decoder_only.attention.multi_head_attention import MultiHeadAttention
from transformer_decoder_only.attention.self_attention_head import SelfAttentionHead
from transformer_decoder_only.config.default import ModelConfig

def test_build_causal_mask_returns_expected_shape_and_dtype() -> None:
    """
    Test that the causal mask has shape `(1, T, T)` and boolean dtype

    The leading dimension is `1` so the mask can broadcast over a batch of
    attention scores with shape `(B, T, T)`.
    """
    
    mask = build_causal_mask(sequence_length=4)

    assert mask.shape == (1, 4, 4)
    assert mask.dtype == torch.bool

def test_build_causal_mask_returns_lower_triangular_allowed_pattern() -> None:
    """
    Test that the causal mask allows current and previous positions only.

    For sequence length 4, the expected pattern is:

        [[ True, False, False, False],
         [ True,  True, False, False],
         [ True,  True,  True, False],
         [ True,  True,  True,  True]]

    In plain language:

    - row 0 can attend to position 0 only
    - row 1 can attend to positions 0 and 1
    - row 2 can attend to positions 0, 1, and 2
    - row 3 can attend to positions 0, 1, 2, and 3
    """

    mask = build_causal_mask(sequence_length=4)

    expected = torch.tensor(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        ],
        dtype=torch.bool,
    )

    assert torch.equal(mask, expected)

def test_build_cause_mask_raises_for_non_integer_sequence_length() -> None:
    """
    Test that `sequence_length` must be an integer.
    """
    with pytest.raises(TypeError, match="sequence_length must be an integer"):
        build_causal_mask(sequence_length=4.0)  # type: ignore[arg-type]

def test_build_causal_mask_raises_for_non_positive_sequence_length() -> None:
    """
    Test that the mask cannot be built for an empty or negative sequence length.
    """
    with pytest.raises(ValueError, match="sequence_length must be greater than 0"):
        build_causal_mask(sequence_length=0)

def test_apply_causal_mask_replaces_blocked_scores_with_negative_infinity() -> None:
    """
    Test that blocked future positions are replaced with `-inf`.

    This is important because the attention softmax is applied after masking.
    Values of `-inf` because probability zero under softmax.
    """

    attention_scores = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ]
    )
    mask = build_causal_mask(sequence_length=3)

    masked_scores = apply_causal_mask(attention_scores, mask)

    expected = torch.tensor(
        [
            [
                [1.0, float("-inf"), float("-inf")],
                [4.0, 5.0, float("-inf")],
                [7.0, 8.0, 9.0],
            ]
        ]
    )

    assert torch.equal(masked_scores, expected)

def test_apply_causal_mask_preserves_allowed_scores() -> None:
    """
    Test that allowed scores are not changed by masking.

    The mask should only affect future positions. Scores on and below the
    diagonal should remain exactly as they were.
    """

    attention_scores = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )
    mask = build_causal_mask(sequence_length=2)

    masked_scores = apply_causal_mask(attention_scores, mask)

    assert masked_scores[0, 0, 0].item() == 1.0
    assert masked_scores[0, 1, 0].item() == 3.0
    assert masked_scores[0, 1, 1].item() == 4.0

def test_apply_causal_mask_raises_for_non_boolean_mask() -> None:
    """
    Test that the mask must be boolean.

    This keeps the meaning of the mask explicit:

    - `True` means allowed
    - `False` means blocked
    """

    attention_scores = torch.zeros((1, 3, 3))
    mask = torch.ones((1, 3, 3), dtype=torch.float32)

    with pytest.raises(TypeError, match="mask must have dtype torch.bool"):
        apply_causal_mask(attention_scores, mask)

def test_apply_causal_mask_raises_for_non_square_attention_scores() -> None:
    """
    Test that attention scores must be square across query and key dimensions.

    For self-attention over one sequence, the score shape should be `(B, T, T)`.
    """

    attention_scores = torch.zeros((2, 3, 4))
    mask = torch.ones((1, 3, 4), dtype=torch.bool)

    with pytest.raises(ValueError, match="square over the sequence dimensions"):
        apply_causal_mask(attention_scores, mask)

def test_self_attention_head_returns_expected_shape() -> None:
    """
    Test that one self-attention head preserves batch and sequence length while
    reducing the feature dimension to `head_dim`.

    With:

    - `embedding_dim = 8`
    - `num_heads = 2`

    each head has:

    - `head_dim = 4`

    So input shape `(B, T, 8)` should produce output shape `(B, T, 4)`
    """
    config = ModelConfig(
        vocab_size=10,
        context_length=5,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )

    head = SelfAttentionHead(config)
    hidden_states = torch.randn((3, 5, 8))

    output = head(hidden_states)

    assert output.shape == (3, 5, 4)

def test_self_attention_head_raises_for_wrong_input_rank() -> None:
    """
    Test that a self-attention head expects hidden states of shape `(B, T, C)`.

    Raw token ids of shape `(B, T)` should not be accepted here.
    """

    config = ModelConfig(
        vocab_size=10,
        context_length=5,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    head = SelfAttentionHead(config)
    token_ids = torch.randint(0, 10, (3, 5), dtype=torch.long)

    with pytest.raises(ValueError, match="rank 3"):
        head(token_ids)

def test_self_attention_head_raises_when_sequence_exceeds_context_length() -> None:
    """
    Test that attention refuses sequences longer than the configured context window.
    """

    config = ModelConfig(
        vocab_size=10,
        context_length=4,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    head = SelfAttentionHead(config)
    hidden_states = torch.randn((3, 5, 8))

    with pytest.raises(ValueError, match="exceeds the configured context length"):
        head(hidden_states)

def test_multi_head_attention_returns_original_embedding_shape() -> None:
    """
    Test that multi-head attention returns shape `(B, T, C)`.

    This is essential because the decoder block will use residual addition:

        x = x + attention_output

    which requires both tensors to have the same shape.
    """

    config = ModelConfig(
        vocab_size=10,
        context_length=5,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    attention = MultiHeadAttention(config)
    hidden_states = torch.randn((3, 5, 8))

    output = attention(hidden_states)

    assert output.shape == hidden_states.shape

def test_multi_head_attention_uses_expected_number_of_heads() -> None:
    """
    Test that the module creates the configured number of self-attention heads.
    """
    
    config = ModelConfig(
        vocab_size=10,
        context_length=5,
        embedding_dim=8,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    attention = MultiHeadAttention(config)

    assert len(attention.heads) == 4

def test_multi_head_attention_raises_for_wrong_final_dimension() -> None:
    """
    Test that multi-head attention rejects hidden states with the wrong feature width.

    If the model expects `embedding_dim = 8`, then the final tensor dimension
    must be 8.
    """

    config = ModelConfig(
        vocab_size=10,
        context_length=5,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    attention = MultiHeadAttention(config)
    hidden_states = torch.randn((3, 5, 12))

    with pytest.raises(ValueError, match="final dimension"):
        attention(hidden_states)