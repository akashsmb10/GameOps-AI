# Fine-tuning guide (educational, optional)

Module: `src/finetuning/lora_demo.py` · install: `pip install -r requirements-ml.txt` · not needed by the platform.

## Concepts in one page
* **Tokens** - text is split into sub-word pieces (BPE). "crashing" may become "crash" + "ing"; each piece has an id.
* **Embeddings** - each token id maps to a learned vector; similar meanings end up close together.
* **Transformer** - a stack of blocks, each with self-attention + a feed-forward network, residual connections and
  layer normalisation.
* **Self-attention** - every token builds a query, key and value vector; attention weight = softmax(q·k/√d) decides how
  much each other token's value contributes. A causal mask stops a token looking at future tokens, which is what makes
  next-token generation possible.
* **Pretraining** - learn to predict the next token on a very large corpus → general language ability.
* **Fine-tuning** - continue training on a small task dataset so the model follows a format or task.
* **LoRA** - freeze the original weight W and learn a low-rank update ΔW = B·A (rank r ≪ d). Only A and B are trained
  (here 3.3% of parameters), adapters are small files, and the base model is shared.
* **RAG vs fine-tuning** - RAG adds *knowledge* at question time with citations and instant updates; fine-tuning changes
  *behaviour/format*. For "what are players complaining about this week", RAG + tools is the right tool: the facts
  change daily and must be cited. Fine-tuning suits fixed output formats (e.g. structured issue tags).

## The task
Input `Review: <text>\nCategories:` → output ` bugs, servers` (or ` none`). Data: synthetic demo reviews with generator
labels, split by the generator's train/validation/test splits (3,000 train, 300 validation, 300 test examples).
The loss is computed only on target tokens (prompt tokens are masked with -100).

## What was run here (verified)
`python -m src.finetuning.lora_demo --model tiny-random --epochs 5` on CPU (39 s):

| | Before | After LoRA |
|---|---|---|
| Validation loss | 7.60 | 5.96 |
| Validation perplexity | 1,997 | 386 |
| Category micro-F1 (80 test examples) | 0.0 | 0.0 |

* Base: randomly initialised GPT-2 (2 layers, 128 dims, 686k parameters) with a BPE tokenizer trained on the data.
* LoRA r=8, alpha=16 on `c_attn` and `c_proj`: 22.5k trainable parameters (3.3%).
* Interpretation: loss fell and the model learned the output format (it now emits "none", the most frequent answer),
  but a random base cannot learn the review→category mapping through low-rank adapters. **This verifies the pipeline,
  not model quality.** A bug found while running it (missing byte-level decoder → outputs like "Ġnone") was fixed.

## Intended run (implemented, NOT verified here)
```bash
python -m src.finetuning.lora_demo --model HuggingFaceTB/SmolLM2-135M-Instruct --epochs 1 --lr 2e-4 --batch 8
```
SmolLM2-135M (Apache-2.0) is ~270 MB; LoRA targets `q_proj`/`v_proj`. Expect minutes on a consumer GPU, longer on CPU.
The script writes `artifacts/finetuning/finetune_results.json` and the adapter to `artifacts/finetuning/lora_adapter/`.
Hugging Face was unreachable in the build environment, so no pretrained run or result is claimed.

## Hardware notes
CPU-only run used ~1 GB RAM. For a 135M model, 8 GB RAM suffices on CPU; a 4-8 GB GPU makes it practical.
