"""
Tests for the next-token prediction dataset used in the educational transform project.

This module is intentionally small, but it plays an important role in making
the training pipeline trustworthy. It helps us lock in the expected behaviour
of the dataset the rest of the model stack starts depending on it.

That matters because the dataset defines the actual supervised learning problem.

If the dataset forms `(x, y)` pairs incorrectly, then:

- the model will be trained on the wrong target
- loss values will still compute, but they will mean the wrong thing
- debugging later componenets becomes much harder

So these tests are designed to check the basic dataset mechanics carefully.

In plain language:

- does the dataset produce the right number of examples?
- does each example have the right shape?
- is the target sequence really just the input shifted by one token?
- does the dataset fail clearly for invalid input?

Notes
-----
- These tests use tiny hand-checkable token sequences.
- The goal is clarity rather than cleverness.

For example, if the token stream is:

    [10, 11, 12, 13, 14]

and `context_length = 3`, then the valid examples are:

    - index 0:
        x = [10, 11, 12]
        y = [11, 12, 13]

    - index 1:
        x = [11, 12, 13]
        y = [12, 13, 14]

This is the core next-token prediction patten that the rest of the project
will rely on.
"""

import pytest
import torch

from transformer_decoder_only.datasets.language_model_dataset import LanguageModelDataset

def test_dataset_length_matches_number_of_valid_sliding_windows() -> None:
    """
    Test that dataset length follows the expected sliding-window formula.

    If the token stream has length `N` and each example uses `context_length`
    input tokens, then the number of valid examples is:

        N - context_length

    This is because each example also needs one extra token to form the shifted
    target sequence.
    """

    token_ids = torch.tensor([10, 12, 12, 13, 14], dtype=torch.long)
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=3)

    assert len(dataset) == 2

def test_first_item_returns_expected_input_and_target_sequences() -> None:
    """
    Test that the first dataset item is sliced correctly.

    For the token stream:

        [10, 11, 12, 13, 14]

    and `context_length = 3`, the first example should be:

        x = [10, 11, 12]
        y = [11, 12, 13]

    In plain language, the target should always be the input shifted forward by
    one position.
    """

    token_ids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=3)

    x, y = dataset[0]

    assert torch.equal(x, torch.tensor([10, 11, 12], dtype=torch.long))
    assert torch.equal(y, torch.tensor([11, 12, 13], dtype=torch.long))

def test_second_item_returns_expected_shifted_window() -> None:
    """
    Test that later dataset items slide forward by one token.

    For the same token stream and context length, the second example should be:

        x = [11, 12, 13]
        y = [12, 13, 14]

    This confirms that the dataset is exposing overlapping windows rather than
    disjoint chunks.    
    """

    token_ids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=3)

    x, y = dataset[1]

    assert torch.equal(x, torch.tensor([11, 12, 13], dtype=torch.long))
    assert torch.equal(y, torch.tensor([12, 13, 14], dtype=torch.long))

def test_input_and_target_have_expected_shape() -> None:
    """
    Test that each dataset item returns tensors of shape `(context_length,)`.

    This matters because later, a DataLoader will batch these examples into
    tensors of shape `(B, T)`, where:

        - `B` is batch size
        - `T` is context length
    """

    tokens_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    dataset = LanguageModelDataset(token_ids=tokens_ids, context_length=4)

    x, y = dataset[0]

    assert x.shape == (4,)
    assert y.shape == (4,)

def test_target_is_input_shifted_by_one_position() -> None:
    """
    Test the central language-modelling property of the dataset.

    The target sequence should always be the input sequence shifted one token
    to the left in time.

    Example:

        x = [1, 2, 3, 4]
        y = [2, 3, 4, 5]

    This is the actual supervised learning signal for next-token prediction.
    """

    token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=4)

    x, y = dataset[0]

    assert torch.equal(y[:-1], x[1:])
    assert y[-1].item() == 5

def test_dataset_accepts_minimum_number_of_tokens_for_one_example() -> None:
    """
    Test that the dataset works when there are exactly enough tokens for one example.

    To form one example of length `T`, we need exactly:

        - `T` input tokens
        - `1` extra token for the shifted target
    
    so the minimum total length is:

        - `T + 1`
    """

    token_ids = torch.tensor([7, 8, 9, 10], dtype=torch.long)
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=3)

    assert len(dataset) == 1

    x, y = dataset[0]
    assert torch.equal(x, torch.tensor([7, 8, 9], dtype=torch.long))
    assert torch.equal(y, torch.tensor([8, 9, 10], dtype=torch.long))

def test_initialisation_raises_type_error_for_non_tensor_input() -> None:
    """
    Test that `tokens_ids` must be provided as a PyTorch tensor.

    The rest of the pipeline operates on tensors, so the dataset should reject
    plain Python lists and other incompatible types.
    """

    with pytest.raises(TypeError, match="token_ids must be a torch.Tensor"):
        LanguageModelDataset(token_ids=[1, 2, 3, 4], context_length=3)  # type: ignore[arg-type]

def test_initialisation_raises_error_for_non_one_dimensional_tensor() -> None:
    """
    Test that the token stream must be one-dimensional.

    This dataset expects a single flat token sequence, not a matrix or higher-rank tensor.
    """

    token_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)

    with pytest.raises(ValueError, match="one-dimensional tensor"):
        LanguageModelDataset(token_ids=token_ids, context_length=2)

def test_initialisation_raises_error_when_context_length_is_not_positive() -> None:
    """
    Test that `context_length` must be greater than zero.

    A non-positive context length would not define a meaningful training example.
    """

    token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    with pytest.raises(ValueError, match="context_length must be greater than 0"):
        LanguageModelDataset(token_ids=token_ids, context_length=0)

def test_initialisation_raises_error_when_not_enough_tokens_are_available() -> None:
    """
    Test that the dataset rejects token streams that are too short.

    If `context_length = 4`, then we need at least 5 tokens total to form one
    valid `(x, y)` pair.
    """

    token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    with pytest.raises(ValueError, match="context_length \\+ 1 tokens"):
        LanguageModelDataset(token_ids=token_ids, context_length=4)

def test_getitem_raises_type_error_for_non_integer_index() -> None:
    """
    Test that dataset indices must be integers.

    This keeps the indexing behaviour explicit and easy to reason about.
    """
    token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=3)

    with pytest.raises(TypeError, match="index must be an integer"):
        dataset["0"]  # type: ignore[index]

def test_getitem_raises_error_for_negative_index() -> None:
    """
    Test that negative indexing is rejected explicitly.

    PyTorch datasets do not need to support Python's negative list-style indexing,
    and rejecting it keeps the behaviour simple and clear for this project.
    """

    token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=3)

    with pytest.raises(IndexError, match="non-negative"):
        dataset[-1]

def test_getitem_raises_error_for_out_of_range_index() -> None:
    """
    Test that indexing beyond the final valid example fails clearly.
    """

    token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=3)

    with pytest.raises(IndexError, match="out of range"):
        dataset[2]