"""
Full decoder-only transformer language model for the educational transformer project.

This module is where the model are assembled into one completed neural
network. It gives the rest of the repository a stable way to talk about:

- token embeddings
- positional embeddings
- stacked decoder blocks
- final layer normalisation
- vocabulary logits
- the full forward pass from token ids to next-token predictions

Keeping the full model in one place makes the project easier to extend because:

- `embeddings` can focus on turning ids and positions into vectors
- `attention` can focus on causal self-attention mechanics
- `blocks` can focus on reusable transformer block structure
- `training` can focus on loss computation and optimisation
- `inference` can focus on autoregressive generation

In plain language:

- this module answers the question:

    "How do all the transformer pieces fit together?"

- it defines the object that will be trained to predict the next character

Notes
-----
- This is a decoder-only transformer.
- "Decoder-only" means the model uses masked self-attention over a single token
  sequence. There is no separate encoder and no cross attention.
- This is the natural architecture for next-token prediction because the model's job is:

    "Given the previous tokens, predict the next token."

- The causal mask inside the attention heads ensures that each token position
  can only use the current and previous positions, not future ones.
- The model takes integer token ids as input:

    `(B, T)`

  and returns vocabulary logits:

    `(B, T, V)`

  where:

  - `B` = batch size
  - `T` = sequence length
  - `V` = vocabulary size

- The logits are not probabilities yet.
- A logit is an unnormalised score. Later:

    - training will pass logits to cross-entropy loss
    - generation will apply softmax to convert logits into probabilities

Full shape flow
---------------
Let:

- `B` = batch size
- `T` = sequence length
- `C` = embedding dimension
- `V` = vocabulary size

Then the forward pass follows this shape flow:

    1. Raw token ids:

        (B, T)

    2. Token embeddings:

        (B, T, C)

    3. Positional embeddings:

        (1, T, C)

    4. Combined hidden states:

        (B, T, C)

    5. After stacked decoder blocks:

        (B, T, C)

    6. After final layer normalisation:

        (B, T, C)

    7. Vocabulary logits:

        (B, T, V)

Example
-------
If:

    batch_size = 16
    context_length = 64
    embedding_dim = 128
    vocab_size = 40

then:

- input token ids have shape `(16, 64)`
- hidden states have shape `(16, 64, 128)`
- output logits have shape `(16, 64, 40)`

In plain language:

- for every sequence in the batch
- for every position in the sequence
- the model outputs one score for every possible vector
"""

import torch
from torch import Tensor
from torch import nn

from transformer_decoder_only.blocks.decoder_block import DecoderBlock
from transformer_decoder_only.config.default import ModelConfig
from transformer_decoder_only.embeddings.positional_embedding import PositionEmbedding
from transformer_decoder_only.embeddings.token_embedding import TokenEmbedding
from transformer_decoder_only.utils.shapes import assert_rank, format_shape

class DecoderOnlyTransformer(nn.Module):
    """
    Small decoder-only transformer language model.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing:

        - `vocab_size`
        - `context_length`
        - `embedding_dim`
        - `num_heads`
        - `num_layers`
        - `dropout`
        - `use_bias`

    Attributes
    ----------
    config : ModelConfig
        Stored model configuration.
    vocab_size : int
        Number of tokens in the vocabulary.
    context_length : int
        Maximum sequence length the model can process.
    embedding_dim : int
        Hidden feature width used throughout the model.
    token_embedding : TokenEmbedding
        Module that converts token ids into learned token vectors.
    positional_embedding : PositionalEmbedding
        Module that creates learned position vectors.
    embedding_dropout : nn.Dropout
        Dropout applied after token and positional embeddings are added.
    blocks : nn.ModuleList
        Stack of decoder blocks.
    final_norm : nn.LayerNorm
        Final layer normalisation applied before the output projection.
    output_projection : nn.Linear
        Linear layer that maps hidden states to vocabulary logits.

    Notes
    -----
    - The model is intentionally small and readable.
    - The forward pass can be thought of as:

        token ids
            -> token vectors
            -> add position vectors
            -> pass through decoder blocks
            -> normalise
            -> project to vocabulary scores

    - The output scores are called logits.
    - Important:

        - logits are not probabilities
        - logits are not token ids
        - logits are scores over the vocabulary

    - During training, each logit vector is compared with the true next token id
      using cross-entropy loss.
    - During generation, the logits for the final position are converted into a
      probability distribution over the next token.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialise the full decoder-only transformer model.

        Parameters
        ----------
        config : ModelConfig
            Model configuration describing the architecture.

        Notes
        -----
        - We validate the configuration immediately so errors appear at model
          construction time rather than deep inside the forward pass.
        - Unlike some earlier modules, this full model requires `vocab_size > 0`
          because it must create:

            - a token embedding table of shape `(V, C)`
            - an output projection shape of `(C, V)`
        """

        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError("config must be an instance of ModelConfig.")

        config.validate()

        if config.vocab_size <= 0:
            raise ValueError(
                "config.vocab_size must be greater than 0 before constructing "
                "the full decoder-only transformer."
            )

        self.config = config
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.embedding_dim = config.embedding_dim

        # Token embedding
        #   - Converts integer token ids into learned vectors
        #
        #       (B, T) -> (B, T, C)
        self.token_embedding = TokenEmbedding(config)

        # Positional embedding
        #   - Creates learned position vectors for positions:
        #
        #     0, 1, 2, ..., T - 1
        #
        #       (B, T) -> (1, T, C)
        self.positional_embedding = PositionEmbedding(config)

        # Dropout after adding token and positional embeddings
        #   - This is common in transformer models. It slightly regularises
        #     the combined representation before it enters the decoder blocks.
        #
        #       (B, T, C) -> (B, T, C)
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Stack of decoder blocks
        #   - If `num_layers = 2`, then the hidden states pass through two full
        #     transformer blocks in sequence.
        #   - Each block preserves shape
        #
        #       (B, T, C) -> (B, T, C)
        self.blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.num_layers)
        )

        # Final layer normalisation
        #   - This stabilises the final hidden states before projecting them to
        #     vocabulary logits.
        #
        #       (B, T, C) -> (B, T, C)
        self.final_norm = nn.LayerNorm(
            normalized_shape=self.embedding_dim,
            elementwise_affine=True,
            bias=config.use_bias,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Run the full decoder-only transformer forward pass.

        Parameters
        ----------
        token_ids : Tensor
            Integer token ids with shape `(B, T)` where:

            - `B` is batch size
            - `T` is sequence length

        Returns
        -------
        Tensor
            Vocabulary logits with shape `(B, T, V)` where:

            - `B` is batch size
            - `T` is sequence length
            - `V` is vocabulary size

        Notes
        -----
        - The model predicts the next token at every position in the input sequence.
        - If the input is:

            [h, e, l, l]

          then the model outputs four logit vectors:

            - logits at position 0 try to predict the token after `h`
            - logits at position 1 try to predict the token after `he`
            - logits at position 2 try to predict the token after `hel`
            - logits at position 3 try to predict the token after `hell`

        - The causal mask inside attention prevents each position from seeing
          future tokens while computing these predictions.

        Full shape flow
        ---------------
        Input:

            token_ids -> `(B, T)`

        Token embedding:

            `(B, T) -> (B, T, C)`

        Positional embedding:

            `(B, T) -> (1, T, C)`

        Add embeddings:

            `(B, T, C) + (1, T, C) -> (B, T, C)`

        Decoder blocks:

            `(B, T, C) -> (B, T, C)`

        Final normalisation:

            `(B, T, C) -> (B, T, C)`

        Output projection:

            `(B, T, C) -> (B, T, V)`
        """
        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor.")
        
        # The full model starts from raw integer token ids
        #   - Expected shape:
        #
        #       (B, T)
        assert_rank(token_ids, expected_rank=2, tensor_name="token_ids")

        if token_ids.dtype != torch.long:
            raise TypeError(
                "token_ids must have dtype torch.long, "
                f"but received dtype {token_ids.dtype}."
            )

        batch_size, sequence_length = token_ids.shape

        # We unpack batch size for readability and teaching, even though this
        # method only needs it later for explicit output-shape validations
        _ = batch_size

        if sequence_length > self.context_length:
            raise ValueError(
                "sequence length exceeds the configured context length. "
                f"Received sequence length {sequence_length}, but the maximum "
                f"supported context length is {self.context_length}."
            )
        
        # Token embeddings give each token id a learned vector
        token_embeddings = self.token_embedding(token_ids)

        # Positional embeddings give each sequence position a learned vector
        #   - The leading dimension is 1 because the same positions are shared
        #     across all items in the batch.
        positional_embeddings = self.positional_embedding(token_ids)

        # Combine token identity and token position
        #   - Addition broadcasts the `(1, T, C)` positional tensor across the
        #     batch dimension.
        #   - This enriches each token vector with positional information about
        #     where it occurs.
        hidden_states = token_embeddings + positional_embeddings

        # Apply dropout after combining embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        # Pass hidden states through the stack of decoder blocks
        #   - The blocks are applied sequentially, so the output of one block becomes
        #     the input to the next.
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # Apply final normalisation before the vocabulary projection
        hidden_states = self.final_norm(hidden_states)

        # Project final hidden states to vocabulary logits
        #   - For each position, this produces one score for every possible token in
        #     the vocabulary.
        logits = self.out_projection(hidden_states)

        expected_shape = (token_ids.shape[0], sequence_length, self.vocab_size)

        if logits.shape != expected_shape:
            raise ValueError(
                "DecoderOnlyTransformer produced logits with the wrong shape. "
                f"Expected {expected_shape}, received {format_shape(logits)}."
            )

        return logits
    
    @property
    def num_parameters(self) -> int:
        """
        Return the total number of trainable parameters in the model.

        Returns
        -------
        int
            Number of parameters whose values will be updated during training.

        Notes
        -----
        - A PyTorch model is made of parameter tensors.
          
          For example:

            - an embedding table might have shape `(V, C)`
            - a linear layer weight might have shape `(out_features, in_features)`
            - a layer norm scale vector might have shape `(C,)`

        - Each tensor contains many individual scalar values.
        - The method `parameter.numel()` counts how many scalar values are inside
          one tensor.
        - This property adds those counts across all trainable parameter tensors.
        - In plain language, it answers:

            "How many learned numbers does this model contain?"

          This is useful because parameter count is a rough measure of model size.
        - More parameters usually mean:

            - more capacity to learn patterns
            - more memory usage
            - more computation per training step
            - more risk of overfitting on a tiny corpus

        Example
        -------
        If a linear layer has a weight matrix with shape:

            (128, 40)

        then that one tensor contains:

            128 * 40 = 5120

        trainable scalar values
        """

        # `self.parameters()` iterates over every parameter tensor registered
        # inside this model, including parameters inside childe modules such as:
        #
        #   - token embeddings
        #   - positional embeddings
        #   - attention projections
        #   - feed-forward layers
        #   - layer normalisation
        #   - output projection
        #
        # `parameter.requires_grad` is True for tensors that the optimiser should
        # update during training.
        #
        # `parameter.numel()` counts the number of scalar values inside that tensor.
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )