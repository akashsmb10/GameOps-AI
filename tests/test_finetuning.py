import pytest

torch = pytest.importorskip("torch")  # optional module: skipped when PyTorch is not installed

from src.finetuning.lora_demo import build_examples, encode  # noqa: E402


class CharTok:
    pad_token_id, eos_token_id = 0, 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 50) for c in text]}


def test_examples_use_generator_splits_and_targets():
    d = build_examples(max_train=50, max_eval=10)
    assert len(d["train"]) == 50 and len(d["validation"]) == 10
    assert all(e["prompt"].startswith("Review: ") and e["prompt"].endswith("Categories:") for e in d["train"])


def test_loss_is_masked_to_target_tokens():
    ids, attn, lab = encode([{"prompt": "abc", "target": " de"}], CharTok())
    assert lab[0, :3].tolist() == [-100, -100, -100]
    assert (lab[0, 3:] != -100).sum() == 4  # " de" (3 chars) + eos
    assert attn.sum() == ids.shape[1]
