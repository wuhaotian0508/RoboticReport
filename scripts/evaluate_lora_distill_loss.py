"""Evaluate LoRA checkpoints on cached distillation examples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from train_style_lora_distill import (
    MOCONVQ,
    build_agent_clean,
    build_gpt,
    load_pretrained_gpt,
)
from style_lora import inject_lora, load_lora_state_dict
import MoConVQCore.Utils.pytorch_utils as ptu


def loss_on_examples(model, examples, device: torch.device) -> tuple[float, int]:
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            bert_feature = example["bert_feature"].to(device)
            bert_mask = example["bert_mask"].to(device)
            latents = example["latents"].to(device)
            idxs = example["idxs"].to(device).long()
            clip_feature = torch.zeros((1, 512), device=device)
            logits, _ = model(latents[:, :-1], idxs, clip_feature, bert_feature, bert_mask)
            pred = logits[:, :, :4, :512].reshape(-1, 512)
            target = idxs.reshape(-1)
            losses.append(float(F.cross_entropy(pred, target).detach().cpu()))
    return sum(losses) / max(len(losses), 1), len(losses)


def checkpoint_sort_key(path: Path):
    if path.stem.endswith("last"):
        return 10**9
    try:
        return int(path.stem.rsplit("epoch", 1)[1])
    except Exception:
        return 10**8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/finetune_lora_200"))
    parser.add_argument("--train-cache", type=Path, default=Path("outputs/finetune_lora_200/distill_cache.pt"))
    parser.add_argument("--val-cache", type=Path, default=Path("outputs/finetune_lora_val/distill_cache.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/lora_train_val_loss.csv"))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    train_examples = torch.load(args.train_cache, map_location="cpu")
    val_examples = torch.load(args.val_cache, map_location="cpu")

    agent, _ = build_agent_clean(args.device)
    ptu.init_gpu(torch.cuda.is_available(), gpu_id=args.device)
    agent.simple_load(str(MOCONVQ / "moconvq_base.data"), strict=True)
    agent.eval()

    checkpoints = sorted(args.checkpoint_dir.glob("style_lora_epoch*.pth"), key=checkpoint_sort_key)
    rows = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["config"]
        model = build_gpt(agent, device)
        load_pretrained_gpt(model, device)
        inject_lora(
            model,
            rank=int(config.get("lora_rank", 8)),
            alpha=float(config.get("lora_alpha", 16.0)),
            dropout=float(config.get("lora_dropout", 0.0)),
            target_prefixes=tuple(str(config.get("target_prefix", "trans_temporal")).split(",")),
        )
        model.to(device)
        load_lora_state_dict(model, checkpoint["lora"])
        train_loss, train_count = loss_on_examples(model, train_examples, device)
        val_loss, val_count = loss_on_examples(model, val_examples, device)
        rows.append(
            {
                "checkpoint": checkpoint_path.name,
                "epoch": checkpoint.get("epoch", ""),
                "train_examples": train_count,
                "train_loss": f"{train_loss:.6f}",
                "val_examples": val_count,
                "val_loss": f"{val_loss:.6f}",
            }
        )
        print(f"{checkpoint_path.name}: train={train_loss:.6f} val={val_loss:.6f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "epoch",
                "train_examples",
                "train_loss",
                "val_examples",
                "val_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
