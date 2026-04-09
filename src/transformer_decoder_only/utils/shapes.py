"""
Tensor-shape helper utilities for the educational decoder-only transformer project.

This module is intentionally small, but it plays an important practical role in
the project. It gives the rest of the repository a stable way to talk about:

- tensor rank
- tensor shape
- shape validation
- clear error messages during debugging

Keeping shape-related helpers in one place makes the project easier to extend because:

- `embeddings` can focus on representation learning rather than manual shape checks
- `attention` can focus on the mecanics of Q, K, V and masking
- `models` can focus on composing layers rather than repeating validation logic
- `tests` can rely on clear, predictable shape error messages

In plain language:

- this module answers the question, "Are these tensors shaped the way we expect?"
- it also helps us print shapes in a readable form when explaining the model

Notes
-----
- Transformer code is full of tensors whose shapes carrying meaning.
- For example:

    - token ids often have shape `(B, T)`
    - embeddings often have shape `(B, T, C)`
    - attention scores often have shape `(B, T, T)`

  where:

    - `B` means batch size
    - `T` means sequence length or context length
    - `C` means embedding dimension

- Shape mistakes are among the most common beginner errors in PyTorch code.
- For that reason, it is useful to have a few small helpers that:

    - make expectations explicit
    - raise errors early
    - produce readable messages

- The goal here is not to build a large validation framework. The goal is to
  build a few small, clear helpers that are easy to use while developing
  the educational model.
"""

from collections.abc import Sequence

from torch import Tensor

def get_shape_tuple(tensor: Tensor) -> tuple[int, ...]:
    """
    Return a tensor's shape as a plain Python tuple.

    Parameters
    ----------
    tensor : Tensor
        Tensor whose shape should be returned.

    Returns
    -------
    tuple[int, ...]
        Shape of the tensor as a tuple of integers.

    Notes
    -----
    - PyTorch already exposes `tensor.shape`, but returning an explicit tuple
      can sometimes be more convenient when building error messages or 
      writing tests.

    Example
    -------
    If `tensor.shape` is:

        torch.Size([16, 64, 128])

    then this function returns:

        (16, 64, 128)
    """

    if not isinstance(tensor, Tensor):
        raise TypeError("tensor must be a torch.Tensor.")
    
    return tuple(tensor.shape)

def format_shape(tensor: Tensor) -> str:
    """
    Format a tensor's shape as a readable string.

    Parameters
    ----------
    tensor : Tensor
        Tensor whose shape should be formatted.

    Returns
    -------
    str
        Human-readable shape string.

    Notes
    -----
    - This helper exists mainly to make debug messages and teaching examples
      slightly easier to read.

    Example
    -------
    If a tensor has shape `(16, 64, 128)`, this function returns:

        "(16, 64, 128)"
    """

    shape = get_shape_tuple(tensor)
    return str(shape)

def assert_rank(tensor: Tensor, expected_rank: int, tensor_name: str = "tensor") -> None:
    """
    Assert that a tensor has the expected number of dimensions.

    Parameters
    ----------
    tensor : Tensor
        Tensor to validate.
    
    expected_rank : int
        Expected number of dimensions.

        For example:

        - rank 1 means shape like `(N,)`
        - rank 2 means shape like `(B, T)`
        - rank 3 means shape like `(B, T, C)`

    tensor_name : str, default="tensor"
        Human-readable name used in error messages.

    Notes
    -----
    - Rank checks are often the first useful sanity check in tensor code.

    Example
    -------
    If token ids should have shape `(B, T)`, then they should have rank 2.

    If a tensor arrives with shape `(B, T, C)` instead, that usually means the
    wrong stage of the pipeline has been passed into the function.
    """

    if not isinstance(tensor, Tensor):
        raise TypeError(
            f"{tensor_name} must be a torch.Tensor."
        )
    
    if not isinstance(expected_rank, int):
        raise TypeError("expected_rank must be an integer.")
    
    if expected_rank < 0:
        raise ValueError("expected_rank must be non-negative.")
    
    actual_rank = tensor.ndim

    if actual_rank != expected_rank:
        raise ValueError(
            f"{tensor_name} must have rank {expected_rank}, "
            f"but received rank {actual_rank} with shape {format_shape(tensor)}."
        )
    
def assert_last_dim(
        tensor: Tensor,
        expected_last_dim: int,
        tensor_name: str = "tensor",
) -> None:
    """
    Assert that a tensor's final dimension has the expected size.

    Parameters
    ----------
    tensor : Tensor
        Tensor to validate.
    expected_last_dim : int
        Expected size of the final dimension.
    tensor_name : str, default="tensor"
        Human-readable name used in error messages.

    Notes
    -----
    - This helper is especially useful in transformer code, where many
      tensors may have unknown batch and sequence dimesnions but a known
      feature width.
    - For example, hidden states often have shape:

        `(B, T, C)`

      where `B` and `T` may vary, but the final dimension `C` should match
      the model embedding dimension.

    Example
    -------
    If hidden states are expected to have embedding width 128, then valid shapes
    include:

        `(4, 64, 128)`
        `(16, 32, 128)`

    but not:

        `(4, 64, 256)`
    """

    if not isinstance(tensor, Tensor):
        raise TypeError(f"{tensor_name} must be a torch.Tensor.")
    
    if not isinstance(expected_last_dim, int):
        raise TypeError("expected_last_dim must be an integer.")
    
    if expected_last_dim <= 0:
        raise ValueError("expected_last_dim must be greater than 0.")
    
    if tensor.ndim == 0:
        raise ValueError(
            f"{tensor_name} must have at least one dimension to check its final "
            f"dimension, but received scalar shape {format_shape(tensor)}."
        )
    
    actual_last_dim = tensor.shape[-1]

    if actual_last_dim != expected_last_dim:
        raise ValueError(
            f"{tensor_name} must have final dimension {expected_last_dim}, "
            f"but received shape {format(tensor)}."
        )
    
def assert_shape(
        tensor: Tensor,
        expected_shape: Sequence[int | None],
        tensor_name: str = "tensor",
) -> None:
    """
    Assert that a tensor matches an expected shape pattern.

    Parameters
    ----------
    tensor : Tensor
        Tensor to validate.

    expected_shape : Sequence[int | None]
        Expected shape pattern.

        Each entry describes one dimension.

        - an integer means that dimension must match exactly
        - `None` means "any size is acceptable here

        This makes it possible to check partially known shapes such as:

        - `(None, None, 128)` for hidden states of shape `(B, T, C)`
        - `(None, 64)` for token-id batches with fixed context length

    tensor_name : str, default="tensor"
        Human-readable name used in error messages.

    Notes
    -----
    - This helper is useful when some dimensions are conceptually important and
      fixed, while others vary naturally.

    Example
    -------
    Suppose hidden states should have shape `(B, T, 128)`, but `B` and `T`
    may vary.

    Then we can check:

        expected_shape = (None, None, 128)

    This means:

    - first dimension: any value is acceptable
    - second dimension: any value is acceptable
    - third dimension: must be 128
    """

    if not isinstance(tensor, Tensor):
        raise TypeError(f"{tensor_name} must be a torch.Tensor.")
    
    if not isinstance(expected_shape, Sequence):
        raise TypeError("expected_shape must be a sequence of integers or None values.")
    
    actual_shape = get_shape_tuple(tensor)

    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"{tensor_name} must have shape pattern {tuple(expected_shape)}, "
            f"but received shape {actual_shape}."
        )
    
    for dimension_index, (actual_dim, expected_dim) in enumerate(
        zip(actual_shape, expected_shape, strict=True)
    ):
        if expected_dim is None:
            continue

        if not isinstance(expected_dim, int):
            raise TypeError(
                "Each value in expected_shape must be either an integer or None."
            )
        
        if expected_dim < 0:
            raise ValueError("Expected shape dimensions must be non-negative or None.")
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"{tensor_name} must have shape pattern {tuple(expected_shape)}, "
                f"but received shape {actual_shape}. "
                f"Dimension {dimension_index} expected {expected_dim} and received {actual_dim}."
            )
        
def assert_same_shape(
        first: Tensor,
        second: Tensor,
        first_name: str = "first",
        second_name: str = "second",
) -> None:
    """
    Assert that two tensors have exactly the same shape.

    Parameters
    ----------
    first : Tensor
        First tensor to compare.
    second : Tensor
        Second tensor to compare.
    first_name : str, default="first"
        Human-readable name for the first tensor.
    second_name : str, default="second"
        Human-readable name for the second tensor.

    Notes
    -----
    - This is especially useful in places where two tensors are meant to align
      position-by-position, such as:

        - model logits and targets after certain reshaping steps
        - residual-connection inputs and outputs
        - paired tensors used in tests

    Example
    -------
    If two hidden-state tensors are both meant to represent the same batch of
    token positions, then they should often have exactly the same shape.
    """

    if not isinstance(first, Tensor):
        raise TypeError(f"{first_name} must be a torch.Tensor.")

    if not isinstance(second, Tensor):
        raise TypeError(f"{second_name} must be a torch.Tensor.")

    first_shape = get_shape_tuple(first)
    second_shape = get_shape_tuple(second)

    if first_shape != second_shape:
        raise ValueError(
            f"{first_name} and {second_name} must have the same shape, "
            f"but received {first_shape} and {second_shape}."
        )