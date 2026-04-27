"""
Tests for the full decoder-only transformer model.

This module verifies that the complete model is wired together correctly before
we move on to training and generation.

That matters because the full model combines many smaller components:

- token embeddings
- positional embeddings
- decoder blocks
- multi-head attention
- feed-forward networks
- final layer normalisation
- vocabulary projection

If the full model is assembled incorrectly, training code may fail later with
less helpful errors. These tests give us a focused checkpoint.

In plain language:

- can we construct the model?
- can it accept tokens?
- does it return logits with the expected shape?
- does it reject invalid input clearly?
- does it expose a sensible trainable parameter count?

Notes
-----
- These are structural tests, not learning-quality tests.
- We are not checking whether the model can produce good text yet.
- We are checking whether the model can perform a valid forward pass.

Shape reminder
--------------
Let:

- `B` = batch size
- `T` = sequence length
- `V` = vocabulary size

The full model should map:

    token ids: `(B, T)`

to:

    logits: `(B, T, V)`

The logits are scores over the vocabulary for every sequence position.
"""

import pytest
import torch

from transformer_decoder_only.config.default import ModelConfig
from transformer_decoder_only.models.decoder_transformer import DecoderOnlyTransformer

def make_tiny_model_config() -> ModelConfig:
    """
    Build a tiny valid model configuration for tests.

    Notes
    -----
    - We keep this model deliberately small so tests run quickly.
    - The only key constraint is that:

        embedding_dim % num_heads == 0

      so that each attention head receives an equal feature width.
    """

    return ModelConfig(
        vocab_size=12,
        context_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )

def test_model_constructs_with_valid_config() -> None:
    """
    Test that the full decoder-only transformer can be constructed.

    This confirms that all child modules can be initialised together:

    - token embedding
    - positional embedding
    - decoder blocks
    - final norm
    - output projection
    """
    
    config = make_tiny_model_config()

    model = DecoderOnlyTransformer(config)

    assert isinstance(model, DecoderOnlyTransformer)

def test_model_rejects_config_with_zero_vocab_size() -> None:
    """
    Test that the full model requires a positive vocabulary size.

    Earlier configuration objects may temporarily allow `vocab_size = 0`
    because the tokeniser has not necessarily read the corpus yet.

    The full model cannot allow that, because it needs to create:

    - token embedding table with shape `(V, C)`
    - output projection with shape `(C, V)`
    """

    config = ModelConfig(
        vocab_size=0,
        context_length=8,
        embedding_dim=8,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="vocab_size must be greater than 0"):
        DecoderOnlyTransformer(config)

def test_forward_returns_logits_with_expected_shape() -> None:
    """
    Test the most important full-model shape contract.

    The model should map:

        token_ids -> logits

    with shape flow:
    
        `(B, T) -> (B, T, V)`

    where:

    - `B` is batch size
    - `T` is sequence length
    - `V` is vocabulary size
    """

    config = make_tiny_model_config()
    model = DecoderOnlyTransformer(config)

    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(3, 5),
        dtype=torch.long,
    )

    logits = model(token_ids)

    assert logits.shape == (3, 5, config.vocab_size)

def test_forward_accepts_sequence_equal_to_context_length() -> None:
    """
    Test that the model accepts a sequence exactly as long as the context window.

    If `context_length = 8`, then sequence length `T = 8` should be valid.
    """
    config = make_tiny_model_config()
    model = DecoderOnlyTransformer(config)

    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, config.context_length),
        dtype=torch.long,
    )

    logits = model(token_ids)

    assert logits.shape == (2, config.context_length, config.vocab_size)

def test_forward_rejects_sequence_longer_than_context_length() -> None:
    """
    Test that the model rejects sequences longer than the configured context window.

    The positional embedding table only contains positions:

        0, 1, ..., context_length - 1

    so longer sequences cannot be represented by this model.
    """

    config = make_tiny_model_config()
    model = DecoderOnlyTransformer(config)

    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, config.context_length + 1),
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="exceeds the configured context length"):
        model(token_ids)

def test_forward_rejects_non_tensor_input() -> None:
    """
    Test that the model expects token ids as a PyTorch tensor.

    This keeps the public interface explicit.
    """
    config = make_tiny_model_config()
    model = DecoderOnlyTransformer(config)

    with pytest.raises(TypeError, match="token_ids must be a torch.Tensor"):
        model([[1, 2, 3]])  # type: ignore[arg-type]

def test_forward_rejects_wrong_rank_input() -> None:
    """
    Test that the model expects token ids with shape `(B, T)`.

    A rank-1 tensor like `(T,)` is missing the batch dimension.
    A rank-3 tensor like `(B, T, C)` is already embedded and therefore belongs
    deeper inside the model, not at the model input.
    """
    config = make_tiny_model_config()
    model = DecoderOnlyTransformer(config)

    token_ids = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(ValueError, match="rank 2"):
        model(token_ids)

def test_forward_rejects_non_long_token_ids() -> None:
    """
    Test that token ids must use integer dtype `torch.long`.

    Embedding layers use token ids as lookup indices, so floating-point values
    are not valid.
    """
    config = make_tiny_model_config()
    model = DecoderOnlyTransformer(config)

    token_ids = torch.ones((2, 4), dtype=torch.float32)

    with pytest.raises(TypeError, match="dtype torch.long"):
        model(token_ids)

def test_forward_rejects_token_ids_outside_vocabulary_range() -> None:
    """
    Test that invalid token ids fail clearly.

    This validation happens inside the token embedding module. We include it in
    the full-model tests because invalid token ids are a common pipeline error.
    """
    config = make_tiny_model_config()
    model = DecoderOnlyTransformer(config)

    token_ids = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, config.vocab_size],
        ],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="outside the vocabulary range"):
        model(token_ids)

def test_num_parameters_returns_positive_integer() -> None:
    """
    Test that the model exposes a positive trainable parameter count.

    This confirms that trainable parameters are registered correctly through
    PyTorch's `nn.Module` machinery.

    In plain language, the model should contain learned numbers that the
    optimiser can update during training.
    """
    config = make_tiny_model_config()
    model = DecoderOnlyTransformer(config)

    assert isinstance(model.num_parameters, int)
    assert model.num_parameters > 0

def test_num_parameters_is_stable_between_calls() -> None:
    """
    Test that reading `num_parameters` does not mutate the model.

    The property should simply count registered parameters. Calling it multiple
    times should return the same value.
    """
    config = make_tiny_model_config()
    model = DecoderOnlyTransformer(config)

    first_count = model.num_parameters
    second_count = model.num_parameters

    assert first_count == second_count