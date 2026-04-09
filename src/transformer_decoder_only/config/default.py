"""
Shared configuration objects for the education decode-only transformer project.

This module is intentionally small, but it will play an important architectural role.
It gives the rest of the repository a stable vocabulary for talking about:

- how large the model should be
- how long each input sequence should be
- how many attention heads and decoder blocks should be used
- how training should be run
- how autoregressive generation should be controlled

Keeping these structures in one place makes the project easier to extend because:

- `src.transformer_decoder_only.tokenisation` can focus on turning text into token ids
- `src.transformer_decoder_only.datasets` can focus on building next-token examples
- `src.transformer_decoder_only.attention` can focus of masking self-attention
- `src.transformer_decoder_only.models` can focus on composing the full model
- `src.transformer_decoder_only.training` can focus on optimisation and evaluation
- `src.transformer_decoder_only.inference` can focus on text generation

In plain language:

- this module describes the "shape of the settings"
- the rest of the codebase describes what to *do* with these settings

Notes
-----
- The classes in this module use `@dataclass(slots=True)`.
- That combination does two useful things:

    1. `@dataclass` automatically generates standard data-container behaviour such
       as an `__init__` method and a readable representation.
    2. `slots=True` keeps the set of allowed attributes fixed to the declared fields.

This is appropriate here because these classes are intended to be:

    - lightweight containers
    - explicit in their structure
    - stable across the rest of the codebase

- In practice, this makes the project a little safer and clearer because these
  objects are not meant to grow arbitrary extra attributes at runtime.

- We also keep validation methods close th the configuration objects.
  This helps mistakes fail early with clear error messages, rather than later
  as confusing tensor-shape failures deep inside the model.
"""

from dataclasses import dataclass, field

@dataclass(slots=True)
class ModelConfig:
    """
    Configuration for the decoder-only transformer model.

    Attributes
    ----------
    vocab_size : int, default=0
        Number of unique tokens in the vocabulary.

        For a character-level model, this is simply the number of unique
        characters discovered in the training corpus.

        We allow this to start at `0` because the tokeniser will usually
        discover the vocabulary only after reading the text.
    
    context_length : int, default=64
        Maximum number of tokens the model can consider at once.

        This is sometimes called the context window or block size.

        If `context_length=64`, then each training example gives the model
        64 positions of input context, and the model learns to predict the
        next token at each of those positions.

    embedding_dim : int, default=128
        Width of the learned vector representation used throughout the model.

        Every token embedding has this width.
        Every positional embedding has this width.
        Every decoder block receives and returns with this width.

    num_heads : int, default=4
        Number of parallel attention heads used in each multi-head attention layer.

        Each head works on a smaller slice of the full embedding dimension.

    num_layers : int, default=2
        Number of stacked decoder blocks.

        More layers usually mean greater modelling capacity, but they also make
        the model slower and a little harder to inspect.

    dropout : float, default=0.1
        Dropout probability used in the model.

        For this first educational implementation, we keep dropout small.

    use_bias : bool, default=True
        Whether linear layers and layner normalisation should include bias terms.

        Keeping this configurable makes later experimentation easier.
    """

    vocab_size: int = 0
    context_length: int = 64
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    use_bias: bool = True

    @property
    def head_dim(self) -> int:
        """
        Width of each individual attention head.

        Notes
        -----
        Multi-head attention splits the full embedding dimension across heads.

        So if:

            embedding_dim = 128
            num_heads = 4

        then each head receives:

            heads_dim = 128 // 4 = 32

        This is important because the query, key, and value projections for a
        single head all work in this smaller subspace.

        Example
        -------
        If the full hidden representation at each token position has shape:

            (B, T, 128)

        then one attention head will usually produce:

        - `Q` with shape `(B, T, 32)`
        - `K` with shape `(B, T, 32)`
        - `V` with shape `(B, T, 32)`

        where 
        
        - `B` is batch size
        - `T` is sequence length
        """

        return self.embedding_dim // self.num_heads
    
    def validate(self) -> None:
        """
        Validate model hyperparameters.

        Notes
        -----
        We validate here so that configuration mistakes are caught early and
        reported clearly.

        For example, if `embedding_dim` is not divisible by `num_heads`, then
        multi-head attention cannot split the representation evenly across heads.

        In plain language, validation answers the question:

            "Are these settings internally consistent enough for the model to work?"
        """

        if self.vocab_size < 0:
            raise ValueError("vocab_size must be non-negative.")
        
        if self.context_length <= 0:
            raise ValueError("context_length must be greater than 0.")
        
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0.")
        
        if self.num_heads <= 0:
            raise ValueError("num_heads must be greater than 0.")
        
        if self.num_layers <= 0:
            raise ValueError("num_layers must be greater than 0.")
        
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0]")
        
        # Multi-head attention only works cleanly if each head receives the 
        # same number of features. That means the full embedding width must be
        # divisible by the number of heads.
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads so that each "
                "attention head receives an equal share of features."
            )
        
        # We deliberately do not raise an error when vocab_size is 0.
        #   - This is normal early in the project because the tokeniser will
        #     often be built after the default configuration object is created.
        #   - Later, once the corpus has been inspected, it will create a model
        #     configuration with the real discovered vocabulary size.
        if self.vocab_size == 0:
            pass

@dataclass(slots=True)
class TrainingConfig:
    """
    Configuration for model training.

    Attributes
    ----------
    batch_size : int, default=16
        Number of sequences processed together in one optimisation step.

        Larger batches usually give smoother gradient estimates, but they also
        require more memory and more work per step.

        for a small CPU-friendly educational model, 16 is a sensible starting point.

    learning_rate : float, default=3e-4
        Step size used by the optimiser when updating parameters.

        If this is too large, training can become unstable.
        If this is too small, training can become unnecessarily slow.

    num_epochs : int, default=10
        Number of full passes over the training data.
    
    eval_interval : int, default=100
        How often to run or print evaluation during training, measured in steps.
    
    eval_batches : int, default=20
        Number of batches to use when estimating validation loss.

        Using a small fixed number keeps evaluation reasonably quick while still
        giving a useful estimate of model quality.

    train_split : float, default=0.9
        Fraction of the token stream used for training.

        The remaining fraction is used for validation.
    
    seed : int, default=5901
        Random seed used to make runs more reproducible.

    device : str, default="cpu"
        Device on which tensors and the model should be placed.

        For the first implementation, CPU is the intended default.
    """

    batch_size: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 10
    eval_interval: int = 100
    eval_batches: int = 20
    train_split: float = 0.9
    seed: int = 5901
    device: str = "cpu"

    def validate(self) -> None:
        """
        Validate training hyperparameters.

        Notes
        -----
        These checks are intentionally simple.

        The goal is not to prevent every unusual experiment.
        The goal is to catch the most common mistakes early, with messages that
        make sense to a beginner.
        """

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be greater than 0.")
        
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be greater than 0.")
        
        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be greater than 0.")
        
        # The split must be strictly between 0 and 1
        #   - If it were 0, there would be no training data.
        #   - If it were 1, there would be no validation data.
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split must be strictly between 0.0 and 1.0.")
        
        # For now, we keep the accepted device names intentionally small and explicit
        #   - This keeps error messages clear and avoids prematurely complicating the
        #     first educational version of the project.
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be either 'cpu' or 'cuda'.")
        
@dataclass(slots=True)
class GenerationConfig:
    """
    Configuration for autoregressive text generation.

    Attributes
    ----------
    max_new_tokens : int, default=200
        Maximum number of new tokens to generate after the prompt.

    temperature : float, default=1.0
        Sampling temperature used to control randomness.

        Lower values make the output distribution sharper.
        Higher values make the output distribution flatter.

        In plain language:

        - low temperature makes the model more conservative
        - high temperature makes the model more varied

    do_sample : bool, default=True
        Whether to sample from the probability distribution over the next token.
        For example, suppose the next token the model predicts has this probability
        distribution:

            "A" -> 0.60
            "B" -> 0.25
            "C" -> 0.10
            "D" -> 0.05
        
        `do_sample=True` means that you randomly choose the next token according
        to the probabilities, so:

            "A" gets picked 60% of the time,
            "B" gets picked 25% of the time, ...

        If set to `False`, generation can instead take the single most likely
        token at each step. This is the "greedy choice". So in the above example,
        "A" is picked because it has the highest probability.
    """

    max_new_tokens: int = 200
    temperature: float = 1.0
    do_sample: bool = True

    def validate(self) -> None:
        """
        Validate generation hyperparameters.

        Notes
        -----
        Generation settings are simpler than training settings, but we still
        validate them so that obviously broken values fail immediately.
        """

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")
        
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than 0.")
        
@dataclass(slots=True)
class ProjectConfig:
    """
    Top-level configuration object for the full project.

    Attributes
    ----------
    model : ModelConfig
        Hyperparameters controlling the decoder-only transformer architecture.

    training : TrainingConfig
        Hyperparameters controlling optimisation and evaluation.

    generation : GenerationConfig
        Hyperparameters controlling autoregressive text generation.

    Notes
    -----
    Grouping the full configuration into one object makes the rest of the
    codebase cleaner because we can pass one structured object around rather
    than passing many unrelated numbers into every function.

    This also gives the project a single obvious home for future additions,
    such as:

    - data-loading settings
    - checkpointing settings
    - experiment metadata
    """

    # `default_factory=...` rather than writing `model=ModelConfig()` directly 
    #   - dataclass fields that hold objects should usually be created afresh 
    #     for each new parent object
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def validate(self) -> None:
        """
        Validate all configuration sections.

        In practice, this gives the rest of the codebase one clear place to call
        before constructing datasets, models, training loops, or generation code.
        """

        self.model.validate()
        self.training.validate()
        self.generation.validate()

def get_default_config() -> ProjectConfig:
    """
    Build and return the default project configuration.

    Returns
    -------
    ProjectConfig
        A validated configuration object containing default mode, training,
        and generation settings.

    Notes
    -----
    We use a small helper function rather than relying only on module-level
    global objects because it keeps configuration creation explicit.

    It also leaves room for later extensions such as:

    - custom experiment presets
    - loading overrides from a file
    - adjustment settings for CPU versus GPU runs

    Example
    -------
    Build the default configuration.

    >>> config = get_default_config()
    >>> config.model.context_length
        64

    In words, this means:

    - create the default grouped configuration object
    - validate that its sections are internally consistent
    - return it ready for use elsewhere in the project
    """

    config = ProjectConfig()
    config.validate()
    return config