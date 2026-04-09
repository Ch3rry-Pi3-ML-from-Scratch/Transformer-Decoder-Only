# Mini Transformer Battle Plan

## Goal

Build a small, educational, decoder-only transformer language model in Python using PyTorch.

The aim is understanding first:
- clear modules
- readable code
- heavy comments
- explicit tensor shapes
- manual implementation of attention internals
- CPU-friendly defaults
- no high-level transformer abstractions

This is not meant to be production-ready. It is meant to make the mechanism understandable.

## Core Design Decisions

### Why decoder-only?

A decoder-only transformer is the simplest useful transformer for language modelling.

It learns:
- how tokens become vectors
- how positions are represented
- how self-attention works
- how causal masking prevents cheating
- how stacked blocks build richer representations
- how next-token prediction trains the model
- how autoregressive generation works

We avoid encoder-decoder complexity so we can focus on the essentials.

### Why character-level tokenisation?

Character-level tokenisation keeps the first build simple.

Advantages:
- very small and transparent vocabulary
- easy to implement from scratch
- no external tokeniser dependency
- easy to inspect encoded sequences

Trade-off:
- longer sequences are needed to express the same information compared with subword tokenisation

For a pedagogical first model, this is the right trade.

### Why PyTorch?

PyTorch gives us:
- tensors
- automatic differentiation
- neural network modules
- optimisers
- loss functions

That lets us implement the real transformer logic ourselves without also building a tensor library and autograd engine from scratch.

We will still implement:
- Q, K, V projections
- scaled dot-product attention
- causal masking
- head concatenation
- output projection

We will not use:
- `torch.nn.Transformer`
- Hugging Face `transformers`
- `nn.MultiheadAttention`

## Target Project Structure

```text
pyproject.toml
data/input.txt
docs/battle-plan.md
src/mini_transformer/__init__.py
src/mini_transformer/config/default.py
src/mini_transformer/tokenisation/char_tokeniser.py
src/mini_transformer/datasets/language_model_dataset.py
src/mini_transformer/embeddings/token_embedding.py
src/mini_transformer/embeddings/positional_embedding.py
src/mini_transformer/attention/causal_mask.py
src/mini_transformer/attention/self_attention_head.py
src/mini_transformer/attention/multi_head_attention.py
src/mini_transformer/mlp/feed_forward.py
src/mini_transformer/blocks/decoder_block.py
src/mini_transformer/models/decoder_transformer.py
src/mini_transformer/training/trainer.py
src/mini_transformer/training/evaluate.py
src/mini_transformer/inference/generate.py
src/mini_transformer/utils/seed.py
src/mini_transformer/utils/shapes.py
src/mini_transformer/main.py
tests/test_tokeniser.py
tests/test_dataset.py
tests/test_attention.py
tests/test_model.py
```

## Recommended Hyperparameters

Initial CPU-friendly defaults:

- `context_length = 64`
- `embedding_dim = 128`
- `num_heads = 4`
- `num_layers = 2`
- `dropout = 0.1`
- `batch_size = 16`
- `learning_rate = 3e-4`
- `num_epochs = 10`
- `max_new_tokens = 200`

Why these values are sensible:
- small enough to train on CPU
- large enough to show meaningful transformer behaviour
- easy to reason about
- `embedding_dim / num_heads = 128 / 4 = 32`, so each head has a clean width

## End-to-End Data Flow

### Training path

1. Read raw text from `data/input.txt`.
2. Build a character vocabulary from unique characters.
3. Convert text into integer token ids.
4. Split token ids into train and validation streams.
5. Build next-token training examples:
   - input `x` is a sequence of length `T`
   - target `y` is the same sequence shifted one step to the left
6. Map token ids to token embeddings.
7. Add positional embeddings.
8. Pass the sequence through stacked decoder blocks.
9. Project final hidden states to vocabulary logits.
10. Compute cross-entropy loss against target token ids.
11. Backpropagate gradients.
12. Update parameters with the optimiser.

### Generation path

1. Start from a prompt.
2. Encode the prompt into token ids.
3. Feed the current context through the model.
4. Take the logits for the final position.
5. Convert logits to probabilities.
6. Sample or choose the next token.
7. Append the new token to the sequence.
8. Trim to the context window if needed.
9. Repeat until the desired number of new tokens has been generated.
10. Decode token ids back into text.

## Build Order

We will build the project in this order so each step depends only on concepts already introduced.

### Phase 1: Configuration and tokenisation

1. `config/default.py`
2. `tokenisation/char_tokeniser.py`

Purpose:
- define hyperparameters
- convert text to integers and back again

### Phase 2: Dataset construction

3. `datasets/language_model_dataset.py`

Purpose:
- turn a token stream into `(x, y)` next-token examples

### Phase 3: Embeddings

4. `embeddings/token_embedding.py`
5. `embeddings/positional_embedding.py`

Purpose:
- represent token identity
- represent token position

### Phase 4: Attention internals

6. `attention/causal_mask.py`
7. `attention/self_attention_head.py`
8. `attention/multi_head_attention.py`

Purpose:
- implement masked self-attention from first principles

Key concepts:
- Q, K, V projections
- dot-product attention scores
- scaling by `sqrt(d_k)`
- masking future positions
- softmax over allowed positions
- weighted sum of values
- head concatenation
- output projection

### Phase 5: Feed-forward and block structure

9. `mlp/feed_forward.py`
10. `blocks/decoder_block.py`

Purpose:
- add non-linear transformation at each position
- combine attention, residual connections, and layer normalisation

### Phase 6: Full model

11. `models/decoder_transformer.py`

Purpose:
- combine embeddings, stacked decoder blocks, final normalisation, and output logits

### Phase 7: Training and evaluation

12. `training/evaluate.py`
13. `training/trainer.py`

Purpose:
- train the model
- estimate validation loss
- understand optimisation steps

### Phase 8: Inference

14. `inference/generate.py`

Purpose:
- generate text autoregressively one token at a time

### Phase 9: Utilities and entry point

15. `utils/seed.py`
16. `utils/shapes.py`
17. `main.py`

Purpose:
- reproducibility
- shape sanity checks
- one script to wire the system together

### Phase 10: Tests

18. `tests/test_tokeniser.py`
19. `tests/test_dataset.py`
20. `tests/test_attention.py`
21. `tests/test_model.py`

Purpose:
- catch mistakes early
- verify shapes and masking logic
- make refactoring safer

## Key Tensor Shapes To Keep In Mind

Let:
- `B` = batch size
- `T` = context length
- `C` = embedding dimension
- `H` = number of heads
- `D` = head dimension where `D = C / H`
- `V` = vocabulary size

### Token ids

- shape: `(B, T)`

### Token embeddings

- shape: `(B, T, C)`

### Positional embeddings

- shape: `(T, C)` or broadcast to `(B, T, C)`

### Single-head Q, K, V

- each shape: `(B, T, D)`

### Attention scores

- shape: `(B, T, T)`

Interpretation:
- for each batch item
- for each query position
- score every key position

### Multi-head concatenation

- shape before output projection: `(B, T, C)`

### Final logits

- shape: `(B, T, V)`

### Cross-entropy reshape

For next-token prediction we usually reshape:
- logits from `(B, T, V)` to `(B * T, V)`
- targets from `(B, T)` to `(B * T)`

This matches what `cross_entropy` expects:
- rows of class scores
- one target class index per row

## Attention Checklist

When implementing attention, explicitly verify each step:

1. Input hidden states arrive with shape `(B, T, C)`.
2. Project to:
   - `Q` with shape `(B, T, D)` for one head
   - `K` with shape `(B, T, D)`
   - `V` with shape `(B, T, D)`
3. Compute raw attention scores with:
   - `Q @ K^T`
   - result shape `(B, T, T)`
4. Scale scores by `sqrt(D)`.
5. Apply causal mask so positions cannot see the future.
6. Apply softmax across the key dimension.
7. Multiply attention weights by `V`.
8. Produce per-head output `(B, T, D)`.
9. Concatenate all heads to get `(B, T, C)`.
10. Apply a final output projection back into model space.

## Training Checklist

The training loop must make the following ideas explicit:

### Example formation

If the token stream is:

```text
[h, e, l, l, o]
```

then a training example might be:
- input: `[h, e, l, l]`
- target: `[e, l, l, o]`

The model learns:
- after `h`, predict `e`
- after `he`, predict `l`
- after `hel`, predict `l`
- after `hell`, predict `o`

### Loss computation

The model outputs logits for every position and every vocabulary item.

If logits have shape `(B, T, V)`, reshape to `(B * T, V)` so cross-entropy can treat each position as a separate classification problem.

Targets reshape from `(B, T)` to `(B * T)`.

### Optimiser step

For each batch:
1. zero old gradients
2. run forward pass
3. compute loss
4. run backpropagation
5. update parameters

### Evaluation

Evaluation should:
- switch the model to evaluation mode
- disable gradient computation
- compute mean loss over validation batches
- switch back to training mode afterwards if needed

## Generation Checklist

Autoregressive generation must be explained carefully:

1. Start with prompt token ids.
2. If the prompt is longer than the context window, keep only the most recent tokens.
3. Run the current context through the model.
4. Take the logits from the last time step only.
5. Optionally divide by temperature.
6. Convert logits to probabilities with softmax.
7. Sample the next token.
8. Append it to the running sequence.
9. Repeat.

Important point:
- the model never generates a whole sentence in one go
- it generates one token at a time, repeatedly feeding its own previous output back in

## Coding Style Rules

- use Python type hints
- use class-based PyTorch modules where appropriate
- keep functions reasonably short
- include British English comments and docstrings
- include heavy inline comments for educational value
- make tensor shape expectations explicit in comments
- raise helpful errors when assumptions are violated

## Practical Milestones

### Milestone 1

Tokeniser and dataset work correctly.

Success criteria:
- text encodes and decodes correctly
- dataset returns aligned `(x, y)` pairs
- tests pass for tokeniser and dataset

### Milestone 2

Single attention head works correctly.

Success criteria:
- causal mask blocks future positions
- attention output shape is correct
- tests pass for masking and attention shape logic

### Milestone 3

Full decoder block works.

Success criteria:
- residual paths preserve shape
- model can process a batch without shape errors

### Milestone 4

Full model trains on tiny text.

Success criteria:
- training loss falls
- validation loss is sensible
- generated text begins to reflect corpus structure

### Milestone 5

Generation works end to end.

Success criteria:
- prompt is encoded
- new tokens are produced
- output can be decoded to text

## Ground Rules For This Build

- no hidden abstractions
- no skipped steps
- no magic tensor operations without explanation
- no use of high-level transformer helper modules
- no premature optimisation
- no production complexity until the educational version is clear

## Immediate Next Step

Start with:
- `src/mini_transformer/config/default.py`

Then continue to:
- `src/mini_transformer/tokenisation/char_tokeniser.py`

These two modules define the core configuration and the vocabulary pipeline the rest of the system depends on.
