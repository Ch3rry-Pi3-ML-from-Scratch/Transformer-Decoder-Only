"""
Training utilities for the educational decoder-only transformer project.

This module is responsible for teaching the model from data. It gives the rest
of the repository a stable way to talk about:

- the optimisation loop
- gradient-based learning
- training loss
- validation loss
- optimiser steps
- training history

Keeping the training loop in one place makes the project easier to extend because:

- `models` can focus on the forward pass
- `datasets` can focus on producing `(inputs, targets)` examples
- `training.evaluate` can focus on validation loss estimation
- `inference` can focus on autoregressive generation
- `main` can focus on wiring the whole project together

In plain language:

- the module answers the question:

    "How does the model actually learn?"

- it repeatedly shows the model training examples and nudges the parameters so
  that correct next-token predictions become more likely

Notes
-----
- The model learns through gradient descent.
- For each training batch, the core sequence is:

    1. run the model forward
    2. compute loss
    3. clear old gradients
    4. backpropagate to compute new gradients
    5. update parameters with the optimiser

- The model outputs logits of shape:

    (B, T, V)

  where:

  - `B` = batch size
  - `T` = sequence length
  - `V` = vocabulary size

- The targets have shape:

    (B, T)

- Cross-entropy treats each token position as a separate classification example,
  so training uses the helper in `training.evaluate` to reshape:

    logit:      (B, T, V) -> (B * T, V)
    targets:    (B, T)    -> (B * T,)

- This module does not define the model architecture itself.
  It defines the process by which the model parameters are updated.
"""

from dataclasses import dataclass, field
from time import perf_counter

import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from transformer_decoder_only.config.default import TrainingConfig
from transformer_decoder_only.training.evaluate import (
    compute_cross_entropy_loss,
    evaluate_loss,
)

@dataclass(slots=True)
class TrainingHistory:
    """
    Container storing training progress over time.

    Attributes
    ----------
    train_losses : list[float]
        Training loss values recorded during optimisation.
    validation_losses : list[float]
        Validation loss values recorded during periodic evaluation.
    evaluation_step : list[int]
        Training step indices at which validation was recorded.

    Notes
    -----
    This object givres us a simple record of how training evolved.

    In plain language, it lets us later ask:

    - did training loss go down?
    - did validation loss go down?
    - at which training steps were evaluations performed?
    """

    train_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    evaluation_steps: list[int] = field(default_factory=list)

def train_one_batch(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    optimiser: Optimizer,
) -> float:
    """
    Train the model on one batch and return the batch loss.

    Parameters
    ----------
    model : nn.Module
        Model being trained.
    inputs : Tensor
        Input token ids with shape `(B, T)`.
    targets : Tensor
        Target token ids with shape `(B, T)`.
    optimiser : Optimizer
        PyTorch optimiser responsible for updating parameters.

    Returns
    -------
    float
        Scalar batch loss as a Python float.

    Notes
    -----
    This function performs one complete optimisation step.

    The sequence is:

    1. forward pass
    2. loss computation
    3. zero old gradients
    4. backward pass
    5. parameter update

    Why do we zero gradients?
    -------------------------
    PyTorch accumulates gradients by default.

    That means if we do not clear them, gradients from previous batches would
    be added to the new batch's gradients, which is not what we want in this
    simple training loop.

    In plain language:

    - make predictions
    - measure how wrong they are
    - compute how each parameter contributed to that error
    - nudge parameters in a better direction
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")

    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor.")

    if not isinstance(targets, Tensor):
        raise TypeError("targets must be a torch.Tensor.")

    if not isinstance(optimiser, Optimizer):
        raise TypeError("optimiser must be a torch.optim.Optimizer.")
    
    # Set training mode so dropout and other training-specific behaviours are active.
    model.train()

    # Step 1: forward pass
    #
    # Shape:
    #
    #   inputs -> (B, T)
    #   logits -> (B, T, V)
    logits = model(inputs)

    # Step 2: compute next-token cross-entropy loss
    #   - The helper reshapes logits and targets internally to the form expected by
    #     `cross_entropy`.
    loss = compute_cross_entropy_loss(logits, targets)

    # Step 3: clear gradients from from the previous batch
    #   - `set_to_none=True` is common efficient option in PyTorch.
    optimiser.zero_grad(set_to_none=True)

    # Step 4: backpropagation
    #   - This computes gradients of the loss with respect to every trainable 
    #     parameter in the model.
    loss.backward()

    # Step 5: parameter update
    #   - The optimiser uses the gradients that were just computed to adjust the
    #     model parameters.
    optimiser.step()

    return float(loss.item())

def train_model(
        model: nn.Module,
        train_dataloader: DataLoader[tuple[Tensor, Tensor]],
        validation_dataloader: DataLoader[tuple[Tensor, Tensor]],
        training_config: TrainingConfig,
) -> TrainingHistory:
    """
    Train the model over multiple epochs.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    train_data_loader : DataLoader[tuple[Tensor, Tensor]]
        Dataloader providing training batches of `(inputs, targets)`.
    validation_dataloader : DataLoader[tuple[Tensor, Tensor]]
        Dataloader providing validation batches of `(inputs, targets)`.
    training_config : TrainingConfig
        Training configuration containing optimiser and evaluation settings.

    Returns
    -------
    TrainingHistory
        Object containing recorded training and validation losses.

    Notes
    -----
    This function is the main training loop for the project.

    It is responsible for:

    - creating the optimiser
    - iterating over epochs
    - iterating over batches
    - training on each batch
    - periodically estimating validation loss
    - storing a record of training progress

    In plain language:

    - keep showing the model examples
    - keep updating its parameters
    - occasionally stop and check how well it performs on held-out data

    Evaluation schedule
    -------------------
    Validation loss is computed every `eval_interval` steps.

    This gives a rough sense of whether the model is:

    - learning useful patterns
    - overfitting
    - improving steadily

    Device handling
    ---------------
    This function moves each batch to `training_config.device` before training
    or evaluation.

    For the first version of this project, that will usually be `"cpu"`.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")

    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be an instance of TrainingConfig.")
    
    training_config.validate()

    # Move the model itself to the requested device before training begins.
    model = model.to(training_config.device)

    # AdamW is a standard and sensible optimiser for transformer training
    #   - It is not the only possible choice, but it is a strong default and keeps
    #     the educational build reasonably close to common practice.
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
    )

    history = TrainingHistory()
    global_step = 0

    training_start_time = perf_counter()

    for epoch_index in range(training_config.num_epochs):
        epoch_start_time = perf_counter()

        for inputs, targets in train_dataloader:
            # Move the current batch to the requested device
            #
            # Shapes:
            #
            #   inputs  -> (B, T)
            #   targets -> (B, T)
            inputs = inputs.to(training_config.device)
            targets = targets.to(training_config.device)

            train_loss = train_one_batch(
                model=model,
                inputs=inputs,
                targets=targets,
                optimiser=optimiser,
            )
            history.train_losses.append(train_loss)

            global_step += 1

            # Periodically estimate validation loss
            #   - We use a limited number of validation batches so evaluation stays
            #     reasonably quick during training.
            if global_step % training_config.eval_interval == 0:
                validation_loss = evaluate_loss(
                    model=model,
                    dataloader=validation_dataloader,
                    device=training_config.device,
                    max_batches=training_config.eval_batches,
                )
                history.validation_losses.append(validation_loss)
                history.evaluation_steps.append(global_step)

        epoch_duration_seconds = perf_counter() - epoch_start_time

        # A short epoch-level progress message is useful during training
        #   - We keep it lightweight and factual.
        print(
            f"Epoch {epoch_index + 1}/{training_config.num_epochs} "
            f"completed in {epoch_duration_seconds:.2f}s"
        )

    total_training_duration_seconds = perf_counter() - training_start_time

    print(f"Training completed in {total_training_duration_seconds:.2f}s")

    return history