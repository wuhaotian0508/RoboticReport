"""Style-caption distillation fine-tuning for MoConVQ's text generator.

This is a practical training bridge for the course project. The downloaded
HumanML3D parquet release contains processed 263-D motion features, not BVH
files that MoConVQ can directly tokenize. This script therefore uses the
pretrained MoConGPT as a teacher: it generates deterministic token targets for
style-focused captions, then fine-tunes a student model to reproduce those
targets.

Use the resulting loss curve/checkpoints as a preliminary adaptation experiment,
not as ground-truth HumanML3D motion-token supervision.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW


ROOT = Path(__file__).resolve().parents[1]
MOCONVQ = ROOT / "MoConVQ"
sys.path.insert(0, str(MOCONVQ))
sys.path.insert(0, str(MOCONVQ / "Script"))

from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer  # noqa: E402
import MoConVQCore.Utils.pytorch_utils as ptu  # noqa: E402
from moconvq_builder import build_agent  # noqa: E402
from transformers import T5EncoderModel, T5Tokenizer  # noqa: E402


GPT_CONFIG = {
    "num_vq": 512,
    "embed_dim": 768,
    "clip_dim": 512,
    "block_size": 52,
    "num_layers": 9,
    "n_head": 8,
    "drop_out_rate": 0.1,
    "fc_rate": 2,
}


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def read_captions(csv_path: Path, max_samples: int | None, seed: int) -> list[str]:
    captions: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            caption = (row.get("caption") or "").strip()
            if caption:
                captions.append(caption)
    random.Random(seed).shuffle(captions)
    if max_samples:
        captions = captions[:max_samples]
    return captions


def build_gpt(agent, device: torch.device) -> Text2Motion_Transformer:
    embeddings = [
        torch.cat(
            [bottle_neck.embedding, torch.zeros_like(bottle_neck.embedding[:2])],
            dim=0,
        ).to(device)
        for bottle_neck in agent.posterior.bottle_neck_list
    ]
    return Text2Motion_Transformer(
        **GPT_CONFIG,
        embeddings=embeddings,
    ).to(device)


def load_pretrained_gpt(model: Text2Motion_Transformer, device: torch.device) -> None:
    state = torch.load(MOCONVQ / "text_generation_GPT.pth", map_location=device)
    model.load_state_dict(strip_module_prefix(state), strict=True)


def text2bert(
    texts: list[str],
    tokenizer: T5Tokenizer,
    encoder: T5EncoderModel,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = encoder(**encoded)
    return output.last_hidden_state, ~encoded["attention_mask"].bool()


def clean_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def moconvq_runtime():
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        sys.argv = [old_argv[0]]
        os.chdir(MOCONVQ)
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


def build_agent_clean(device_id: int):
    with moconvq_runtime():
        return build_agent(gpu=device_id)


@torch.no_grad()
def make_teacher_example(
    teacher: Text2Motion_Transformer,
    caption: str,
    tokenizer: T5Tokenizer,
    encoder: T5EncoderModel,
    device: torch.device,
    max_length: int,
) -> dict[str, torch.Tensor | str] | None:
    bert_feature, bert_mask = text2bert([caption], tokenizer, encoder, device)
    clip_feature = torch.zeros((1, 512), device=device)
    latents, idxs = teacher.sample(
        clip_feature,
        bert_feature,
        bert_mask,
        if_categorial=False,
        max_length=max_length,
    )
    if latents.numel() == 0 or idxs.numel() == 0:
        return None

    token_rows = idxs.view(-1, 4)
    time_steps = min(latents.shape[1], token_rows.shape[0])
    if time_steps < 2:
        return None

    return {
        "caption": caption,
        "bert_feature": bert_feature[:, : bert_feature.shape[1]].detach().cpu(),
        "bert_mask": bert_mask[:, : bert_mask.shape[1]].detach().cpu(),
        "latents": latents[:, :time_steps].detach().cpu(),
        "idxs": token_rows[:time_steps].unsqueeze(0).detach().cpu(),
    }


def build_cache(args: argparse.Namespace, device: torch.device) -> list[dict[str, torch.Tensor | str]]:
    captions = read_captions(Path(args.train_csv), args.max_samples, args.seed)
    if not captions:
        raise RuntimeError(f"No captions found in {args.train_csv}")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")

    agent, _ = build_agent_clean(args.device)
    ptu.init_gpu(torch.cuda.is_available(), gpu_id=args.device)
    agent.simple_load(str(MOCONVQ / "moconvq_base.data"), strict=True)
    agent.eval()

    teacher = build_gpt(agent, device)
    load_pretrained_gpt(teacher, device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    tokenizer = T5Tokenizer.from_pretrained("t5-large", resume_download=True)
    encoder = T5EncoderModel.from_pretrained("t5-large", resume_download=True).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    examples: list[dict[str, torch.Tensor | str]] = []
    for i, caption in enumerate(captions, start=1):
        example = make_teacher_example(
            teacher, caption, tokenizer, encoder, device, args.max_length
        )
        if example is not None:
            examples.append(example)
        print(f"cache {i}/{len(captions)} kept={len(examples)} caption={caption[:80]}")

    Path(args.cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, args.cache_path)
    print(f"saved cache: {args.cache_path} ({len(examples)} examples)")
    del teacher, tokenizer, encoder, agent
    clean_cuda()
    return examples


def load_or_build_cache(args: argparse.Namespace, device: torch.device):
    cache_path = Path(args.cache_path)
    if cache_path.exists() and not args.rebuild_cache:
        examples = torch.load(cache_path, map_location="cpu")
        print(f"loaded cache: {cache_path} ({len(examples)} examples)")
        return examples
    return build_cache(args, device)


def train_student(args: argparse.Namespace, device: torch.device, examples) -> None:
    if not examples:
        raise RuntimeError("No cached examples available for training.")

    agent, _ = build_agent_clean(args.device)
    ptu.init_gpu(torch.cuda.is_available(), gpu_id=args.device)
    agent.simple_load(str(MOCONVQ / "moconvq_base.data"), strict=True)
    agent.eval()

    student = build_gpt(agent, device)
    load_pretrained_gpt(student, device)
    student.train()
    del agent
    clean_cuda()

    optimizer = AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.csv"
    global_step = 0

    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "loss", "caption"])
        writer.writeheader()
        for epoch in range(args.epochs):
            random.Random(args.seed + epoch).shuffle(examples)
            for example in examples:
                bert_feature = example["bert_feature"].to(device)
                bert_mask = example["bert_mask"].to(device)
                latents = example["latents"].to(device)
                idxs = example["idxs"].to(device).long()
                clip_feature = torch.zeros((1, 512), device=device)

                input_latents = latents[:, :-1]
                logits, _ = student(input_latents, idxs, clip_feature, bert_feature, bert_mask)
                pred = logits[:, :, :4, :512].reshape(-1, 512)
                target = idxs.reshape(-1)
                loss = F.cross_entropy(pred, target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

                global_step += 1
                loss_value = float(loss.detach().cpu())
                writer.writerow(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "loss": loss_value,
                        "caption": example["caption"],
                    }
                )
                f.flush()
                if global_step % args.log_every == 0:
                    print(f"epoch={epoch} step={global_step} loss={loss_value:.4f}")

            ckpt_path = output_dir / f"style_adapter_epoch{epoch}.pth"
            torch.save(
                {"model": student.state_dict(), "epoch": epoch, "step": global_step},
                ckpt_path,
            )
            print(f"saved checkpoint: {ckpt_path}")

    torch.save({"model": student.state_dict(), "step": global_step}, output_dir / "style_adapter_last.pth")
    print(f"done: log={log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/style_subset_small/style_train_small.csv")
    parser.add_argument("--output-dir", default="outputs/finetune_distill")
    parser.add_argument("--cache-path", default="outputs/finetune_distill/distill_cache.pt")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if args.train_only:
        examples = torch.load(args.cache_path, map_location="cpu")
    else:
        examples = load_or_build_cache(args, device)

    if args.build_cache_only:
        return 0

    train_student(args, device, examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
