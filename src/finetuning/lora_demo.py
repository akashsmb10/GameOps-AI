"""Educational LoRA fine-tuning demo (optional; independent of the main platform).

Task: "structured issue tagging" - given a review, generate `categories: <list>` (the complaint
categories). Data: the SYNTHETIC demo reviews and their generator labels (MIT, produced by this
repository), split by the generator's train/validation/test splits.

Two model options:
  --model <hf-id or local path>   a small pretrained causal LM, e.g. HuggingFaceTB/SmolLM2-135M-Instruct
                                  (Apache-2.0, ~270 MB). This is the intended use. Requires downloading the
                                  weights once; NOT executed in the build environment (Hugging Face blocked).
  --model tiny-random             a 2-layer GPT-2 built from a config with a tokenizer trained on the data.
                                  Runs on CPU in minutes. It verifies the full pipeline (tokenisation, LoRA
                                  injection, training, adapter saving, before/after evaluation, inference).
                                  Because the base model is randomly initialised it is NOT a language model
                                  in any useful sense; its scores only show that training changes behaviour.

Usage:
  python -m src.finetuning.lora_demo --model tiny-random --epochs 3
  python -m src.finetuning.lora_demo --model HuggingFaceTB/SmolLM2-135M-Instruct --epochs 1   (GPU recommended)
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import pandas as pd

from src.config import CATEGORIES, get_settings

SPECIFIC = [c for c in CATEGORIES if c != "other"]
PROMPT = "Review: {text}\nCategories:"


def build_examples(max_train: int = 3000, max_eval: int = 300, seed: int = 0) -> dict[str, list[dict]]:
    s = get_settings()
    reviews = pd.read_csv(s.raw_dir / "demo_reviews.csv", dtype={"review_id": str}).drop_duplicates("review_id")
    labels = pd.read_parquet(s.raw_dir / "labels_synthetic.parquet")
    df = reviews.merge(labels, on="review_id")
    df = df[df["language"] == "english"]
    df["target"] = df[SPECIFIC].apply(lambda r: ", ".join(c for c in SPECIFIC if r[c]) or "none", axis=1)
    out = {}
    for split, n in (("train", max_train), ("validation", max_eval), ("test", max_eval)):
        d = df[df["split"] == split].sample(min(n, (df["split"] == split).sum()), random_state=seed)
        out[split] = [{"prompt": PROMPT.format(text=t), "target": " " + y} for t, y in zip(d["review_text"], d["target"], strict=True)]
    return out


def _tiny_model(texts: list[str], out_dir: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()  # without it, generated text keeps byte-level markers such as "Ġ"
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=2000, special_tokens=["<unk>", "<pad>", "<eos>"]))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    cfg = GPT2Config(vocab_size=hf_tok.vocab_size, n_positions=256, n_embd=128, n_layer=2, n_head=4,
                     bos_token_id=hf_tok.eos_token_id, eos_token_id=hf_tok.eos_token_id)
    hf_tok.save_pretrained(out_dir / "tokenizer")
    return GPT2LMHeadModel(cfg), hf_tok, ["c_attn", "c_proj"]


def _pretrained(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    targets = ["q_proj", "v_proj"] if any("q_proj" in n for n, _ in model.named_modules()) else ["c_attn"]
    return model, tok, targets


def encode(batch: list[dict], tok, max_len: int = 192):
    """Loss only on the target tokens: prompt tokens are masked with -100."""
    import torch
    ids, labels = [], []
    for ex in batch:
        p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        t = tok(ex["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        x = (p + t)[:max_len]
        y = ([-100] * len(p) + t)[:max_len]
        ids.append(x)
        labels.append(y)
    n = max(len(x) for x in ids)
    pad = tok.pad_token_id
    input_ids = torch.tensor([x + [pad] * (n - len(x)) for x in ids])
    attn = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids])
    lab = torch.tensor([y + [-100] * (n - len(y)) for y in labels])
    return input_ids, attn, lab


def eval_loss(model, data, tok, bs: int = 32) -> float:
    import torch
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data), bs):
            ids, attn, lab = encode(data[i:i + bs], tok)
            out = model(input_ids=ids, attention_mask=attn, labels=lab)
            n = int((lab != -100).sum())
            tot, cnt = tot + float(out.loss) * n, cnt + n
    return tot / max(cnt, 1)


def generate(model, tok, prompt: str, max_new: int = 16) -> str:
    import torch
    model.eval()
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def set_f1(data, model, tok, n: int = 80) -> float:
    """Micro-F1 between generated and true category sets on the first n examples."""
    tp = fp = fn = 0
    for ex in data[:n]:
        pred = {c.strip() for c in generate(model, tok, ex["prompt"]).split(",") if c.strip() in SPECIFIC}
        true = {c.strip() for c in ex["target"].split(",") if c.strip() in SPECIFIC}
        tp, fp, fn = tp + len(pred & true), fp + len(pred - true), fn + len(true - pred)
    return 2 * tp / max(1, 2 * tp + fp + fn)


def main(argv=None) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny-random")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=3000)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    out_dir = get_settings().artifacts_dir / "finetuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_examples(a.max_train)
    if a.model == "tiny-random":
        model, tok, targets = _tiny_model([e["prompt"] + e["target"] for e in data["train"]], out_dir)
    else:
        model, tok, targets = _pretrained(a.model)
    base_params = sum(p.numel() for p in model.parameters())
    before = {"val_loss": eval_loss(model, data["validation"], tok)}
    before["val_perplexity"] = math.exp(before["val_loss"])
    before["test_set_f1"] = set_f1(data["test"], model, tok)

    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM",
                      fan_in_fan_out=a.model == "tiny-random")
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    t0, history = time.perf_counter(), []
    for ep in range(a.epochs):
        model.train()
        order = torch.randperm(len(data["train"])).tolist()
        for i in range(0, len(order), a.batch):
            ids, attn, lab = encode([data["train"][j] for j in order[i:i + a.batch]], tok)
            loss = model(input_ids=ids, attention_mask=attn, labels=lab).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        history.append({"epoch": ep + 1, "val_loss": eval_loss(model, data["validation"], tok)})
    after = {"val_loss": history[-1]["val_loss"], "val_perplexity": math.exp(history[-1]["val_loss"]),
             "test_set_f1": set_f1(data["test"], model, tok)}
    model.save_pretrained(out_dir / "lora_adapter")
    sample = data["test"][0]
    result = {"model": a.model, "verified_here": a.model == "tiny-random",
              "caveat": ("tiny-random is a randomly initialised 2-layer GPT-2: results only demonstrate that the LoRA pipeline "
                         "trains and changes behaviour; they say nothing about pretrained-LLM quality.") if a.model == "tiny-random" else "",
              "task": "review -> complaint categories (synthetic demo labels)", "train_examples": len(data["train"]),
              "eval_examples": len(data["validation"]), "base_parameters": base_params, "trainable_lora_parameters": trainable,
              "trainable_share": round(trainable / base_params, 4), "lora": {"r": a.rank, "alpha": 2 * a.rank, "targets": targets},
              "epochs": a.epochs, "lr": a.lr, "before": before, "after": after, "history": history,
              "train_seconds": round(time.perf_counter() - t0, 1),
              "hardware": f"{platform.processor() or platform.machine()} CPU, torch {torch.__version__}, cuda={torch.cuda.is_available()}",
              "example": {"prompt": sample["prompt"], "target": sample["target"].strip(), "generated": generate(model, tok, sample["prompt"])}}
    (out_dir / "finetune_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
