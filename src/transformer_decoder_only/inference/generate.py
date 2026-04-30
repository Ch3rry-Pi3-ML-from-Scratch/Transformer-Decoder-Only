"""
Autoregressive text-generation utilities for the educational decoder-only
transformer project.

This module is responsible for using a trained model to create new text. It
gives the rest of the repository a stable way to talk about:

- prompts
- autoregressive generation
- context-window cropping
- next-token logits
- temperature scaling
- sampling versus greedy decoding
- decoding generated token ids back into readable text

Keeping generation logic in one place makes the project easier to extend because:

- `models` can focus on producing logits
- `training` can focus on optimisation
- `tokenisation` can focus on text-to-id conversion
- `inference` can focus on the repeated next-token loop

In plain language:

- this module answers the question:

    "Given some starting text, how does this model continue it?"

The important idea is that a decoder-only language model does not generate a
whole paragraph in one forward pass.

Instead, it repeats a small loop:

    1. look at the current context
    2. predict scores for the next token
    3. choose one next token
    4. append that token to the context
    5. repeat

For a character-level model, each "token" is one character.

Example
-------
Suppose the prompt is:

    "The "

and `max_new_tokens = 5`.

Generation might proceed conceptially like this:

    current text: "The "
    model predicts next character: "c"

    current text: "The c"
    model predicts next character: "a"

    current text: "The ca"
    model predicts next character: "t"

    current text: "The cat"
    model predicts next character: " "

    current text: "The cat "
    model predicts next character: "s"

Final generated text:

    "The cat s"

The actual model works with token ids rather than strings, so the real process is:

    prompt text
        -> tokeniser.encode(...)
        -> tensor of token ids
        -> model logits
        -> sampled next token id
        -> tokeniser.decode(...)

Notes
-----
- The model returns logits with shape `(B, T, V)`.

  Here:

  - `B` = batch size
  - `T` = sequence length
  - `V` = vocabulary size

- During generation we use only the logits from the final time step:

        logits[:, -1, :]

  because only the last position predicts the next token after the full current
  context.

- If the generated sequence grows longer than the model's context window, we keep
  only the most recent tokens when calling the model.

  For example, if `context_length = 64` and the running sequence has 100 tokens,
  then the model sees only the final 64 tokens for the next prediction.

- This is normal for fixed-context decoder-only models.
"""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch import nn

from transformer_decoder_only.config.default import GenerationConfig
from transformer_decoder_only.tokenisation.char_tokeniser import CharTokeniser

def _infer_model_device(model: nn. Module) -> torch.device:
    """
    Infer the device currently used by a model.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters should be inspected.

    Returns
    -------
    torch.device
        Device for the first model parameter.

    Notes
    -----
    - Most PyTorch models have parameters. A decoder-only transformer certainly
      does, because it contains embedding tables, linear layers, and layer norms.
    - This helper exists so generation can place input tensors on the same device
      as the model.

    In plain language:

    - if the model lives on CPU, create CPU input tensors
    - if the model lives on CUDA, create CUDA input tensors
    """

    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Cannot infer device because the model has no parameters.") from exc
    
def _get_context_length(model: nn.Module) -> int:
    """
    Read the maximum context length from a model.

    Parameters
    ----------
    model : nn. Module
        Model used for generation.

    Returns
    -------
    int
        Maximum number of tokens the model can process at once.

    Notes
    -----
    - The `DecoderOnlyTransformer` exposes `context_length` directly. We avoid
      importing that concrete class here so the generation helper stays usable with
      compatible model objects in future experiments.
    - A compatible model must therefore expose:

        model.context_length

      and it must be a positive integer.
    """

    context_length = getattr(model, "context_length", None)

    if not isinstance(context_length, int):
        raise TypeError(
            "model must expose an integer `context_length` attribute for generation."
        )
    
    if context_length <= 0:
        raise ValueError("model.context_length must be greater than 0.")

    return context_length

def _validate_prompt_token_ids(prompt_token_ids: Sequence[int]) -> None:
    """
    Validate prompt token ids before generation.

    Parameters
    ----------
    prompt_token_ids : Sequence[int]
        Token ids used as the initial generation context.

    Notes
    -----
    - The model expects at least one token because its forward pass receives a
      tensor with shape `(B, T)`, and generation needs a final time step from which
      to read next-token logits.
    - An empty prompt would produce `T=0`, which does not define a meaningful
      "last position".
    """

    if not isinstance(prompt_token_ids, Sequence):
        raise TypeError("prompt_token_ids must be a sequence of integers.")
    
    if len(prompt_token_ids) == 0:
        raise ValueError("prompt_token_ids must contain at least one token.")
    
    for token_id in prompt_token_ids:
        if not isinstance(token_id, int):
            raise TypeError("Every prompt token id must be an integer.")
        
        if token_id < 0:
            raise ValueError("Prompt token ids must be non-negative.")
        
def generate_token_ids(
    model: nn.Module,
    prompt_token_ids: Sequence[int],
    generation_config: GenerationConfig,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """
    Generate token ids autoregressively from an initial prompt.

    Parameters
    ----------
    model : nn.Module
        Trained decoder-only language model.

        The model is expected to accept token ids with shape `(B, T)` and return
        logits with shape `(B, T, V)`.

    prompt_token_ids : Sequence[int]
        Initial token ids used as a starting context.

    generation_config : GenerationConfig
        Generation settings controlling:

        - number of new tokens
        - temperature
        - sampling versus greedy decoding

    device : torch.device | str | None, default=None
        Device on which generation should run.

        If `None`, this function infers the device from the model parameters.

    generator : torch.Generator | None, default=None
        Optional PyTorch random generator used when sampling.

        Passing a seeded generator makes sampled generation easier to reproduce.

    Returns
    -------
    list[int]
        Prompt token ids followed by newly generated token ids.

    Notes
    -----
    - This function works at token-id level. It does not know anything about text.
      That is deliberate:

        - tokenisation belongs to `CharTokeniser`
        - generation belongs to this function
        - decoding back to text happens in `generate_text`

    Shape flow inside the loop
    --------------------------
    Suppose the running sequence currently has 20 token ids and the model has
    `context_length = 8`.

    We crop to the most recent 8 ids:

        context_ids -> shape `(8,)`
    
    Then we add a batch dimension:

        input_ids -> shape `(1, 8)`

    The model returns:

        logits -> shape `(1, 8, V)`

    We take only the last position:

        next_token_logits -> shape `(1, V)`

    Then we either:

    - sample one token id from the probability distribution, or
    - choose the highest-scoring token id greedily

    The selected id has shape:

        next_token_id -> shape `(1, 1)`

    Finally, we append that single id to the running sequence and repeat.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")

    if not isinstance(generation_config, GenerationConfig):
        raise TypeError(
            "generation_config must be an instance of GenerationConfig."
        )
    
    generation_config.validate()
    _validate_prompt_token_ids(prompt_token_ids)

    context_length = _get_context_length(model)

    if device is None:
        device = _infer_model_device(model)

    device = torch.device(device)

    # Store the previous training/evaluation mode
    #   - This means that generation does not leave the model in a 
    #     surprising state for the caller.
    was_training = model.training

    # Generation should use evaluation behaviour:
    #
    #   - dropout disabled
    #   - no parameter estimates
    #   - no gradient graph
    model.eval()

    generated_token_ids = list(prompt_token_ids)

    try:
        with torch.no_grad():
            for _ in range(generation_config.max_new_tokens):
                # Keep only the most recent tokens that fit inside the model's
                # fixed context window.
                # - If the full generated sequence is shorter than the context
                #   length, this simply keeps the whole sequence.
                context_token_ids = generate_token_ids[-context_length:]

                # Convert the Python list into rank-2 tensor because the model
                # expects a batch dimension
                #
                # Shape:
                #
                #   context_token_ids list length T
                #       -> tensor shape `(1, T)`
                input_ids = torch.tensor(
                    [context_token_ids],
                    dtype=torch.long,
                    device=device,
                )

                # Run the model over the current context
                #
                # Shape:
                #
                #   input_ids -> `(1, T)`
                #   logits    -> `(1, T, V)`
                logits = model(input_ids)

                if not isinstance(logits, Tensor):
                    raise TypeError("model must return a torch.Tensor of logits.")

                if logits.ndim != 3:
                    raise ValueError(
                        "model must return logits with shape `(B, T, V)`, "
                        f"but received shape {tuple(logits.shape)}."
                    )     
                
                # We only need the logits at the final position, because that is
                # the position that predicts the token after the full context
                #
                # Shape:
                #
                #   logits[:, -1, :] -> `(1, V)`
                next_token_logits = logits[:, -1, :]

                # Temperature controls how sharp or flat the distribution is
                #
                # - temperature < 1 makes high-scoring tokens more dominant
                # - temperature > 1 spreads probability more evenly
                #
                # We validated earlier that temperature is greater than zero.
                next_token_logits = (
                    next_token_logits / generation_config.temperature
                )

                if generation_config.do_sample:
                    # Convert logits into probabilities over the vocabulary
                    #
                    # Shape:
                    #
                    #   probabilities -> `(1, V)`
                    probabilities = torch.softmax(next_token_logits, dim=-1)

                    # Sample one token id from the probability distribution.
                    #
                    # Shape:
                    #
                    #   next_token_id -> `(1, 1)`
                    next_token_id = torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generator,
                    )
                else:
                    # Greedy decoding chooses the single highest-scoring token
                    #
                    # `keepdim=True` keeps the shape as `(1, 1)`, matching the
                    # sampled path above.
                    next_token_id = torch.argmax(
                        next_token_logits,
                        dim=-1,
                        keepdim=True,
                    )

                # Convert the one-token tensor back into a plain Python integer
                # before appending it to the running list
                generated_token_ids.append(int(next_token_id.item()))

    finally:
        # Restore the model's original mode even if generation raises an error
        if was_training:
            model.train()
        else:
            model.eval()

    return generated_token_ids

def generate_text(
    model: nn.Module,
    tokeniser: CharTokeniser,
    prompt: str,
    generation_config: GenerationConfig,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """
    Generate readable text from a string prompt.

    Parameters
    ----------
    model : nn.Module
        Trained decoder-only language model.

    tokeniser : CharTokeniser
        Character-level tokeniser used to encode the prompt and decode output ids.

        Important:
        The tokeniser should be the same one used to build the model vocabulary.

    prompt : str
        Starting text for generation.

    generation_config : GenerationConfig
        Generation settings.

    device : torch.device | str | None, default=None
        Device on which generation should run.

    generator : torch.Generator | None, default=None
        Optional seeded generator for reproducible sampling.

    Returns
    -------
    str
        Prompt plus generated continuation.

    Example
    -------
    A typical call after training might look like:

        config = GenerationConfig(max_new_tokens=200, temperature=0.8)
        text = generate_text(
            model=model,
            tokeniser=tokeniser,
            prompt="ROMEO:",
            generation_config=config,
        )

    In plain language, this function performs the full inference path:

        raw prompt text
            -> prompt token ids
            -> generated token ids
            -> decoded generated text

    Notes
    -----
    - This function will raise an error if the prompt contains a character that is
      not in the tokeniser vocabulary.
    - That is expected for this first educational version. Since the model is
      character-level and has no unknown-token fallback, it can only process
      characters that existed in the training corpus.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")

    if not isinstance(tokeniser, CharTokeniser):
        raise TypeError("tokeniser must be an instance of CharTokeniser.")

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")

    if prompt == "":
        raise ValueError("prompt must not be empty.")

    prompt_token_ids = tokeniser.encode(prompt)

    generated_token_ids = generate_token_ids(
        model=model,
        prompt_token_ids=prompt_token_ids,
        generation_config=generation_config,
        device=device,
        generator=generator,
    )

    return tokeniser.decode(generated_token_ids)