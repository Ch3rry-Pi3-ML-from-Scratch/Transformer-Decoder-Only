"""
Random seed utilities for the educational decoder-only transformer project.

This module is intentionally small, but it plays an important practical role in
this project. It gives the rest of the repository a stable way to talk about:

- reproducibility
- random number generation
- repeatable experiments
- predictable debugging

Keeping seed logic in one place makes the project easier to extend because:

- `training` can focus on optimisation rather than seed mechanics
- `inference` can focus on generation rather than random-state setup
- `tests` can rely on repeatable behaviour where appropriate
- future experiment scripts can reuse the same seeding rules

In plain language:

- this module answers the question, 

    "How do we make random behaviour more repeatable?"

- it gives us a small, explicit place to control randomness across the project

Notes
-----
- Deep-learning code often involves randomness in several places, for example:

    - parameter initialisation
    - batch shuffling
    - dropout
    - sampling during text generation

- If we do not control randomness at all, then repeated runs can behave quite
  differently, which makes debugging harder.
- Setting a seed does not magically make every run identical in every possible
  environment, but it usually makes behaviour much more repeatable.
- In this project we mainly care about:

    - Python's built-in random module
    - PyTorch random number generation

- We also provide a helper for building a seeded `torch.Generator`.
  This is useful when later code wants an explicit generator object rather than
  relying only on the global random state.
"""

import random

import torch

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for the project.

    Parameters
    ----------
    seed : int
        Integer seed used to initialise the random number generators.
    deterministic : bool, default=False
        Whether to ask PyTorch for more deterministic behaviour where possible.

        If `True`, PyTorch will prefer determinism over performance in places
        where that trade-off is relevant.

    Notes
    -----
    - This function sets seeds for:

        - Python's built-in `random` module
        - PyTorch on CPU
        - PyTorch on CUDA devices, if CUDA is available
    
    - In plain language, this function says:

        "Please make the main random parts of this project start from a known state."

    Example
    -------
    Set the project seed before constructing the model and training loop.

    >>> set_seed(5901)

    This does not remove all possible sources of variation, but it usually makes 
    repeated runs much easier to compare.
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    
    # Seed Python's built-in random module
    #   - This affects code that uses functions such as:
    #
    #       random.random()
    #       random.shuffle()
    #       random.choice()
    random.seed(seed)

    # Seed PyTorch's CPU random number generator
    #   - This affects many common operations such as:
    #
    #       - random parameter initialisation
    #       - dropout masks
    #       - sampling from probability distributions
    torch.manual_seed(seed)

    # If CUDA is available, seed all visible CUDA devices as well
    #   - This matters only when running on GPU, but including it here makes the
    #     helper a little more future-proof
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode asks PyTorch to prefer repeatable algorithms where possible
    #   - This can be useful for debugging, but it may reduce performance and
    #     some operations may become unavailable under strict determinism.
    if deterministic:
        torch.use_deterministic_algorithms(True)

        # These backend flags matter mainly for CUDA workloads
        #   - They are included here so the helper remains useful if the project is
        #     later run on a GPU rather than only on CPU.
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def make_torch_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """
    Build and return a seeded PyTorch generator.

    Parameters
    ----------
    seed : int
        Integer seed used to initialise the generator.
    device : str, default="cpu"
        Device on which the generator should operate.

        For the first version of this project, `"cpu"` is the normal choice.

    Returns
    -------
    torch.Generator
        A PyTorch generator whose random state has been seeded explicitly.

    Notes
    -----
    - A `torch.Generator` is useful when you want randomness to be controlled by
      an explicit object rather than only the global seed setting.
    - This can be useful for things such as:
        
        - reproducible dataset shuffling
        - reproducible token sampling during generation
        - experiments where different generators should be controlled separately

    Example
    -------
    Build a CPU generator seeded with 5901.

    >>> generator = make_torch_generator(5901)
    >>> isinstance(generator, torch.Generator)
        True
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda',")
    
    # Create a generator object tied to the requested device
    generator = torch.Generator(device=device)

    # Seed the generator explicitly so that code using this generator can be
    # reproduced independently of other random operations elsewhere.
    generator.manual_seed(seed)

    return generator 