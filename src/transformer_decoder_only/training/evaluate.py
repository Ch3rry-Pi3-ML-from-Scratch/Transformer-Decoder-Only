"""
Evaluation utilities for the educational decoder-only transformer project.

This module is intentionally focused and small. It gives the rest of the 
repository a stable way to talk about:

- measuring validation loss
- evaluating without updating model parameters
- temporarily switching the model into evaluation mode
- reshaping logits and targets for cross-entropy loss

Keeping evaluation logic in one place makes the project easier to extend because:

- `training.trainer` can focus on optimisation steps
- `model.decoder_transformer` can focus on forward computation
- `datasets` can focus on providing `(x, y)` examples
- future experiment scripts can reuse the same validation-loss calculation

In plain language:

- this module answers the question,

    "How well is the model doing on held-out data?"

- it computes loss without changing the model

Notes
-----
- Evaluation is different from training.
- During training:

    - gradients are tracked
    - dropout is active
    - optimiser steps update parameters

- During evaluation:

    - gradients are not needed
    - dropout should be disabled
    - parameters should not be updated

- PyTorch provides two key tools for this:

    1. `model.eval()`
       Switches modules such as dropout into evaluation behaviour.

    2. `torch.no_grad()`
       Stops PyTorch from building a computation graph for backgropagation.

- This module computes average cross-entropy loss over a limited number of batches.
- The model returns logits with shape:

        (B, T, V)

  where:

    - `B` = batch size
    - `T` = sequence length
    - `V` = vocabulary size

- In language modelling, each token position is one classification example. So we 
  reshape:

        (B, T, V) -> (B * T, V)
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

def compute_cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute next-token cross-entropy loss from model logits and target token ids.

    Parameters
    ----------
    logits : Tensor
        Vocabulary logits with shape `(B, T, V)` where:

        - `B` is batch size
        - `T` is sequence length
        - `V` is vocabulary size

    targets : Tensor
        Target token ids with shape `(B, T)`.

    Returns
    -------
    Tensor
        Scalar loss tensor.

    Notes
    -----
    The model outputs one logit vector for every token position.

    If:

        logits.shape == (B, T, V)

    then each of the `B * T` positions is a separate classification problem
    over `V` possible vocabulary items.

    PyTorch cross-entropy expects class scores in shape:

        (N, V)

    and target class ids in shape:

        (N,)

    Therefore we flatten the batch and sequence dimensions together:

        logits  -> (B * T, V)
        targets -> (B * T,)

    In plain language:

    - each token position becomes one training/evaludation example
    - each example asks, "Which vocabulary item is the correct next token?"
    """

    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a torch.Tensor.")

    if not isinstance(targets, Tensor):
        raise TypeError("targets must be a torch.Tensor.")

    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape `(B, T, V)`, "
            f"but received shape {tuple(logits.shape)}."
        )

    if targets.ndim != 2:
        raise ValueError(
            "targets must have shape `(B, T)`, "
            f"but received shape {tuple(targets.shape)}."
        )

    batch_size, sequence_length, vocab_size = logits.shape

    if targets.shape != (batch_size, sequence_length):
        raise ValueError(
            "targets must match the batch and sequence dimensions of logits. "
            f"logits shape: {tuple(logits.shape)}, "
            f"targets shape: {tuple(targets.shape)}."
        )

    if targets.dtype != torch.long:
        raise TypeError(
            "targets must have dtype torch.long for cross-entropy loss, "
            f"but received dtype {targets.dtype}."
        )
    
    # Flatten the batch and time dimensions into one dimension
    #
    # Before:
    #
    #   logits -> (B, T, V)
    #
    # After:
    #
    #   logits_flat -> (B * T, V)
    #
    # Each row is one classification problem over the vocabulary.
    logits_flat = logits.reshape(batch_size * sequence_length, vocab_size)

    # Flatten targets in the same order
    #
    # Before:
    #   targets -> (B, T)
    #
    # After:
    #   targets_flat -> (B * T,)
    #
    # Each entry is the correct vocabulary index for the corresponding row in
    # `logits_flat`.
    targets_flat = targets.reshape(batch_size * sequence_length)

    return F.cross_entropy(logits_flat, targets_flat)

def evaluate_loss(
    model: nn.Module,
    dataloader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device | str,
    max_batches: int | None = None,
) -> float:
    """
    Estimate average loss over batches from a dataloader.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.

        For this project, this will usually be a `DecoderOnlyTransformer`.
    
    dataloader : Iterable[tuple[Tensor, Tensor]]
        Iterable producing `(inputs, targets)` batches.

        Expected shapes:

        - inputs:   `(B, T)`
        - targets:  `(B, T)`

    device : torch.device | str
        Device on which evaluation should run, usually `"cpu"` for this project.

    max_batches : int | None, default=None
        Optional maximum number of batches to evaluate.

        If `None`, the full dataloader is evaluated.

    Returns
    -------
    float
        Average cross-entropy loss over the evaluated batches.

    Notes
    -----
    This function does not update model parameters.

    The evaluation loop follows this pattern:

    1. remember whether the model was previously in training mode
    2. switch to evaluation mode with `model.eval()`
    3. disable gradient tracking with `torch.no_grad()`
    4. compute loss over batches
    5. restore the model's previous mode

    Restoring the previous mode is useful because this function may be called
    from inside a training loop. If the model was training before evaluation,
    it should go back to training afterwards.

    In plain language:

    - briefly pause training behaviour
    - measure validation loss
    - put the model back the way it was
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")

    if max_batches is not None:
        if not isinstance(max_batches, int):
            raise TypeError("max_batches must be an integer or None.")

        if max_batches <= 0:
            raise ValueError("max_batches must be greater than 0 when provided.")
        
    # Store the current mode so we can restore it afterwards.
    #   - `model.training` is True when the model is in training mode and False 
    #     when it is in evaluation mode.
    was_training = model.training

    # Evaluation mode disables training-specific behaviour such as dropout.
    model.eval()

    total_loss = 0.0
    num_batches = 0

    try:
        # We do not need gradients during evaluation
        #   - This saves memory and computation because PyTorch does not need to
        #     build a graph for backpropagation.
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                # Move both inputs and targets to the requested device
                #   - For this project, that will usually be CPU.
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)
                loss = compute_cross_entropy_loss(logits, targets)

                # `loss` is a scalar tensor
                #   - `.item()` converts it into a plain Python float so we can
                #     accumulate it without keeping unnecessary tensor references.
                total_loss += loss.item()
                num_batches += 1

    finally:
        # Restore the model's previous mode even if evaluation raises an error
        #   - This is defensive and keeps the helper safer to call from training code.
        if was_training:
            model.train()
        else:
            model.eval()

    if num_batches == 0:
        raise ValueError("Cannot evaluate loss because no batches were processed.")

    return total_loss / num_batches