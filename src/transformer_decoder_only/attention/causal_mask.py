"""
Causal-mask utilities for the educational decoder-only transformer project.

This module is intentionally small, but it introduces one of the defining ideas
of decoder-only language models. It gives the rest of the repository a stable
way to talk about:

- which token positions are allowed to attend to which other positions
- how the model is prevented from seeing future tokens
- how next-token prediction remains a valid learning problem
- how masking is applied before the attention softmax

Keeping causal-mask logic in one place makes the project easier to extend because:

- `attention` can focus on Q, K, V and attention-weight calculation
- `models` can assume the mask follows one consistent convention
- `tests` can verify masking behaviour separately from the attention head itself
- future variants can change mask behaviour in one place if needed

In plain language:

- this module answers the question,

    "What is each position allowed to look at?"

- In a decoder-only model, positions may look backwards, but not forwards

Notes
-----
- The word "causal" means the model must respect time/order.
- When predicting the next token at position `t`, the model should only use:

    - tokens at earlier positions
    - the current token

  It should not use tokens from future positions.
- This matters because during training, we already know the full target sequence.
  Without the causal mask, the model could cheat by looking ahead at future tokens.
- For a sequence length of 4, the allowed attention pattern is:

      query position 0: can attend to key positions 0
      query position 1: can attend to key positions 0, 1
      query position 2: can attend to key positions 0, 1, 2
      query position 3: can attend to key positions 0, 1, 2, 3

- As a boolean matrix, using `True` to mean "allowed", this is:

      [[ True, False, False, False],
       [ True,  True, False, False],
       [ True,  True,  True, False],
       [ True,  True,  True,  True]]

- This matrix has shape `(T, T)` where:

    - rows correspond to query positions
    - columns correspond to key positions

- Later, attention scores will have shape `(B, T, T)`.
  PyTorch broadcasting lets us apply a mask of shape `(1, T, T)` across the
  full batch.
"""

import torch
from torch import Tensor

def build_causal_mask(
        sequence_length: int,
        device: torch.device | str | None = None,
) -> Tensor:
    """
    Build a boolean causal attention mask.

    Parameters
    ----------
    sequence_length : int
        Length of the token sequence.

        This is usually called `T` in transformer shape notation.

    device : torch.device | str | None, default=None
        Device on which the mask should be created.

        If `None`, PyTorch creates the mask on the default device, usually CPU.

    Returns
    -------
    Tensor
        Boolean mask with shape `(1, T, T)` where:

        - `True` means the attention connection is allowed
        - `False` means the attention connection is blocked

    Notes
    -----
    - The returned shape includes a leading singleton batch dimension:

        (1, T, T)

      rather than just:

        (T, T)

      This is useful because the attention scores usually have shape:

        (B, T, T)

      and the mask can then be broadcast across all `B` batch items.

      Example
      -------
    For `sequence_length = 4`, this function returns a mask equivalent to:

        [[[ True, False, False, False],
          [ True,  True, False, False],
          [ True,  True,  True, False],
          [ True,  True,  True,  True]]]

    In plain language:

    - row 0 can see only column 0
    - row 1 can see columns 0 and 1
    - row 2 can see columns 0, 1, and 2
    - row 3 can see columns 0, 1, 2, and 3
    """

    if not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer.")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0.")

    # Start with an all-ones square matrix of shape `(T, T)`.
    #
    # At this point, every position is marked as allowed.
    #
    # Example for T = 4:
    #
    #   [[1, 1, 1, 1],
    #    [1, 1, 1, 1],
    #    [1, 1, 1, 1],
    #    [1, 1, 1, 1]]
    all_connections = torch.ones(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=device,
    )

    # Keep only the lower triangle, including the diagonal.
    #
    # `torch.tril` means "triangular lower".
    #
    # This produces the causal pattern:
    #
    #   [[1, 0, 0, 0],
    #    [1, 1, 0, 0],
    #    [1, 1, 1, 0],
    #    [1, 1, 1, 1]]
    #
    # In attention language:
    #
    # - rows are query positions
    # - columns are key positions
    # - values above the diagonal are future positions and must be blocked
    mask = torch.tril(all_connections)

    # Add a leading singleton dimension so the mask can broadcast over a batch.
    #
    # Before:
    #   (T, T)
    #
    # After:
    #   (1, T, T)
    #
    # This matches attention scores shaped like `(B, T, T)`.
    mask = mask.unsqueeze(0)

    return mask

def apply_causal_mask(attention_scores: Tensor, mask: Tensor) -> Tensor:
    """
    Apply a causal mask to raw attention scores.

    Parameters
    ----------
    attention_scores : Tensor
        Raw attention scores with shape `(B, T, T)`.

        These are usually produced by:

            Q @ K.transpose(-2, -1)

    mask : Tensor
        Boolean causal mask with shape `(1, T, T)` or `(B, T, T)`.

        `True` means the attention connection is allowed.
        `False` means the attention connection is blocked.

    Returns
    -------
    Tensor
        Masked attention scores with the same shape as `attention_scores`.

        Positions that are not allowed are replaced with `-inf`, so that after
        softmax they receive probability 0.

    Notes
    -----
    - Masking is applied before softmax.
    - The reason is important:

        - softmax turns scores into probabilities
        - blocked positions should receive probability zero
        - setting their scores to `-inf` before softmax achieves that

    Example
    -------
    Suppose one row of attention scores is:

        [2.0, 5.0, 1.0, 7.0]

    and the mask for that row is:

        [True, True, False, False]

    After masking, the row becomes approximately:

        [2.0, 5.0, -inf, -inf]

    Then softmax assigns probability only to the first two positions.
    """

    if not isinstance(attention_scores, Tensor):
        raise TypeError("attention_scores must be a torch.Tensor.")

    if not isinstance(mask, Tensor):
        raise TypeError("mask must be a torch.Tensor.")

    if attention_scores.ndim != 3:
        raise ValueError(
            "attention_scores must have shape `(B, T, T)`, "
            f"but received shape {tuple(attention_scores.shape)}."
        )

    if mask.ndim != 3:
        raise ValueError(
            "mask must have shape `(1, T, T)` or `(B, T, T)`, "
            f"but received shape {tuple(mask.shape)}."
        )

    if mask.dtype != torch.bool:
        raise TypeError("mask must have dtype torch.bool.")

    batch_size, query_length, key_length = attention_scores.shape
    mask_batch_size, mask_query_length, mask_key_length = mask.shape

    if query_length != key_length:
        raise ValueError(
            "attention_scores must be square over the sequence dimensions, "
            f"but received shape {tuple(attention_scores.shape)}."
        )

    if mask_query_length != query_length or mask_key_length != key_length:
        raise ValueError(
            "mask sequence dimensions must match attention_scores. "
            f"attention_scores shape: {tuple(attention_scores.shape)}, "
            f"mask shape: {tuple(mask.shape)}."
        )

    if mask_batch_size not in {1, batch_size}:
        raise ValueError(
            "mask batch dimension must be either 1 or match attention_scores "
            f"batch size {batch_size}, but received {mask_batch_size}."
        )

    if mask.device != attention_scores.device:
        raise ValueError(
            "mask and attention_scores must be on the same device. "
            f"attention_scores device: {attention_scores.device}, "
            f"mask device: {mask.device}."
        )
    
    # Replace blocked positions with negative infinity
    #   - Why negative infinity?
    #     Because softmax turns very negative values into probabilities
    #     very close to 0.
    #   - This means future positions cannot receive attention probability.
    #   - `~mask` means invert the mask, so if the mask had been:
    #
    #       [True, True, False, False]
    #
    #     Inverting it would produce:
    #
    #       [False, False, True, True]
    #
    #     Then the `masked_fill` says:
    #
    #       "Take attention scores, and wherever `~mask` is true, replace with `-inf`."
    masked_attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

    return masked_attention_scores