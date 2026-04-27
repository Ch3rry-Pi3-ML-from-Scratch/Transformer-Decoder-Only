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


"""