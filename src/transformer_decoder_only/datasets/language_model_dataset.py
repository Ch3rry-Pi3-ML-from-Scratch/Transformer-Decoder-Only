"""
Dataset utilities for next-token prediction in the educational decoder-only
transformer project.

This module is intentionally small, but it plays a very important role in the
training pipeline. It gives the rest of the repository a stable way to talk
about:

- how one long token stream becomes many training examples
- how input sequences and target sequences are aligned
- how next-token prediction is constructed from raw token ids

Keeping dataset logic in one place makes the project easier to understand because:

- `tokenisation` can focus on text-to-id conversion
- `models` can focus on transforming token ids into logits
- `training` can focus on optimisation rather than sample construction
- `tests` can verify that the learning problem is being built correctly

In plain language:

- this module answers the question, "What should one training example look like?"
- it turns a single long sequence of token ids into many `(x, y)` pairs

Notes
-----
- This project trains a decoder-only language model with next-token prediction.
- That means each training example contains:

    - an input sequence `x`
    - an output sequence `y`

  where `y` is just `x` shifted one position to the left.

- For example, if the token stream is:

    [h, e , l, l, o]

  then one training example could be:

    x = [h, e, l, l]
    y = [e, l, l, o]

- In plain language, the model sees:

    - `h` and tries to predict `e`
    - `he` and tries to predict `l`
    - `hel` and tries to predict `l`
    - `hell` and tries to predict `o`

- This is the core supervised learning signal for an autoregressive language model.
- The dataset returns PyTorch tensors because the model and training loop will
  operate on tensors rather than Python lists.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

@dataclass(slots=True)
class LanguageModelDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Dataset of fixed-length next-token prediction examples.

    Attributes
    ----------
    token_ids : Tensor
        One-dimensional tensor containing the full token stream.

        Shape:

            `(N,)`

        where `N` is the total number of token ids available.

    context_length : int
        Length of each input sequence returned by the dataset.

        Each example contains:

        - an input tensor `x` of shape `(context_length,)`
        - a target tensor `y` of shape `(context_length,)`

        where `y` is the input shifted one token forwards.

    Notes
    -----
    - The dataset does not randomly sample positions by itself. Instead, it
      exposes overlapping sliding-window examples.
    - For example, if the token stream is:

        [10, 11, 12, 13, 14]

      and `context_length = 3`, then the examples are:

        - index 0:
            x = [10, 11, 12]
            y = [11, 12, 13]

        - index 1:
            x = [11, 12, 13]
            y = [12, 13, 14]

      so each dataset item asks the model to predict the next token at every
      position in the context window.
    - This overlapping structure is very common in simple language-modelling
      datasets because it makes efficient use of a small corpus.

    Example
    -------
    Build a dataset from a token stream.

    >>> token_ids = torch.tensor([1, 0, 2, 0, 2, 0], dtype=torch.long)
    >>> dataset = LanguageModelDataset(token_ids=token_ids, context_length=4)
    >>> x, y = dataset[0]

    Then:

        - `x` is `[1, 0, 2, 0]`
        - `y` is `[0, 2, 0, 2]`

    In plain language, the target is always the next-token-shiften version of
    the input.
    """

    token_ids: Tensor
    context_length: int

    def __post_init__(self) -> None:
        """
        Validate dataset inputs immediately after initialisation.

        Notes
        -----
        - This method exists to catch mistakes early and clearly.
        - In particular, we want to ensure that:
            - `token_ids` is a PyTorch tensor
            - it is one-dimensional
            - it uses integer token_ids
            - there are enough tokens to form at least one `(x, y)` example
        """
        if not isinstance(self.token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor.")
        
        if self.token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional tensor.")
        
        if self.token_ids.dtype != torch.long:
            raise TypeError("token_ids must be have dtype torch.long.")
        
        if self.context_length <= 0:
            raise ValueError("context_length must be greater than 0.")
        
        # To form one training example of length T, we need T input tokens
        # and T target tokens shifted by one position.
        #   - In practice, that means we need at least:
        #
        #       context_length + 1
        #
        #     total tokens in the underlying token stream.
        if len(self.token_ids) < self.context_length + 1:
            raise ValueError(
                "token_ids must contain at least context_length + 1 tokens "
                "to form one next-token prediction example."
            )
        
    def __len__(self) -> int:
        """
        Return the number of training examples available in the dataset.

        Returns
        -------
        int
            Number of sliding-window `(x, y)` pairs that can be formed.

        Notes
        -----
        - If the full token stream has length `N`, and each example needs:

            - `context_length` input tokens
            - plus 1 extra token so the target can be shifted forward

          then the number of valid staring positions is:

            N - context_length

        Example
        -------
        If:

            N = 10
            context_length = 4

        then the valid start indices are:

            0, 1, 2, 3, 4, 5
        
        which gives:

            10 - 4 = 6 examples
        """

        return len(self.token_ids) - self.context_length
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Return one next-token prediction training example.

        Parameters
        ----------
        index : int
            Starting position of the sliding window.

        Returns
        -------
        tuple[Tensor, Tensor]
            A pair `(x, y)` where:

            - `x` has shape `(context_length,)`
            - `y` has shape `(context_length,)`

            and `y` is `x` shifted one token to the left in time.

        Notes
        -----
        - Suppose the token stream is:

            [10, 11, 12, 13, 14, 15]

          and `context_length = 4`.

          Then at `index = 0`:
            
            x = [10, 11, 12, 13]
            y = [11, 12, 13, 14]

          At `index = 1`:

            x = [11, 12, 13, 14]
            y = [12, 13, 14, 15]

          In plain language, every position in `x` is paired with the token that
          comes immediately after it in `y`

        Tensor shapes
        -------------
        If `context_length = T`, then:

            - `x` has shape `(T,)`
            - `y` has shape `(T,)`

        Later, a DataLoader will batch several examples together so that the
        model receives tensors of shape:

            (B, T)

        where `B` is batch size.  
        """

        if not isinstance(index, int):
            raise TypeError("index must be an integer.")
        
        if index < 0:
            raise IndexError("index must be non-negative.")
        
        if index >= len(self):
            raise IndexError("index is out of range for this dataset.")
        
        # The input window starts at `index` and spans `context_length` tokens
        #
        #   - Shape
        #       x -> (context_length,)
        x = self.token_ids[index : index + self.context_length]

        # The target window is the same slice shifted one position to the right
        #
        #   - Shape
        #       y -> (context_length,)
        #
        #   - This shift is the entire point of next-token prediction.
        y = self.token_ids[index + 1 : index + self.context_length + 1]

        return x, y
        
