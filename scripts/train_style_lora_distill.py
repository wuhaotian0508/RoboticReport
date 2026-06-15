"""Parameter-efficient LoRA distillation for MoConVQ style adaptation.

This script keeps the pretrained MoConVQ text generator frozen and trains only
low-rank LoRA updates on selected language-conditioning transformer linears.
Targets are pseudo motion tokens distilled from the pretrained teacher, matching
the practical constraint that the available HumanML3D parquet data is not
direct MoConVQ token supervision.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW


ROOT = Path(__file__).resolve().parents[1]
MOCONVQ = ROOT / "MoConVQ"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MOCONVQ))
sys.path.insert(0, str(MOCONVQ / "Script"))
sys.path.insert(0, str(MOCONVQ / "ModifyODESrc"))
sys.path.insert(0, str(MOCONVQ / "diff-quaternion" / "TorchRotation"))


def configure_windows_dll_paths() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    import torch as _torch

    candidates = [
        Path(_torch.__file__).resolve().parent / "lib",
        MOCONVQ / "ModifyODESrc",
        MOCONVQ / "diff-quaternion" / "TorchRotation",
    ]
    for path in candidates:
        if path.exists():
            os.add_dll_directory(str(path))


configure_windows_dll_paths()

from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer  # noqa: E402
import MoConVQCore.Utils.pytorch_utils as ptu  # noqa: E402
from moconvq_builder import build_agent  # noqa: E402
from transformers import T5EncoderModel, T5Tokenizer  # noqa: E402

from style_lora import inject_lora, load_lora_state_dict, lora_state_dict  # noqa: E402
from humanml_motion_proxy import (  # noqa: E402
    DEFAULT_SELECTION_METRICS,
    HumanMLMotionStore,
    read_style_rows,
    select_best_candidate,
)


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
    return Text2Motion_Transformer(**GPT_CONFIG, embeddings=embeddings).to(device)


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
    teacher_samples: int = 1,
    target_metrics: dict[str, float] | None = None,
    candidate_metric_fn=None,
    selection_metrics: tuple[str, ...] = DEFAULT_SELECTION_METRICS,
    teacher_categorical_sampling: bool = False,
    teacher_top_k: int = 50,
    teacher_temperature: float = 1.0,
    sample_seed_base: int | None = None,
) -> dict[str, torch.Tensor | str] | None:
    bert_feature, bert_mask = text2bert([caption], tokenizer, encoder, device)
    clip_feature = torch.zeros((1, 512), device=device)
    candidates = []
    for sample_idx in range(max(1, teacher_samples)):
        if sample_seed_base is not None:
            torch.manual_seed(sample_seed_base + sample_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed_base + sample_idx)
        latents, idxs = teacher.sample(
            clip_feature,
            bert_feature,
            bert_mask,
            if_categorial=teacher_categorical_sampling,
            max_length=max_length,
            top_k=teacher_top_k,
            temperature=teacher_temperature,
        )
        if latents.numel() == 0 or idxs.numel() == 0:
            continue
        metrics = None
        if target_metrics is not None and candidate_metric_fn is not None:
            metrics = candidate_metric_fn(latents, sample_idx)
        candidates.append({"latents": latents, "idxs": idxs, "metrics": metrics or {}, "sample_idx": sample_idx})

    if not candidates:
        return None

    if target_metrics is not None and candidate_metric_fn is not None and len(candidates) > 1:
        selected = select_best_candidate(candidates, target_metrics, selection_metrics)
    else:
        selected = candidates[0]

    latents = selected["latents"]
    idxs = selected["idxs"]
    if latents.numel() == 0 or idxs.numel() == 0:
        return None

    token_rows = idxs.view(-1, 4)
    time_steps = min(latents.shape[1], token_rows.shape[0])
    if time_steps < 2:
        return None

    example = {
        "caption": caption,
        "bert_feature": bert_feature.detach().cpu(),
        "bert_mask": bert_mask.detach().cpu(),
        "latents": latents[:, :time_steps].detach().cpu(),
        "idxs": token_rows[:time_steps].unsqueeze(0).detach().cpu(),
    }
    if target_metrics is not None:
        example["humanml_metrics"] = target_metrics
        example["teacher_samples"] = int(max(1, teacher_samples))
        example["selected_sample_idx"] = int(selected.get("sample_idx", 0))
        example["selection_score"] = float(selected.get("selection_score", 0.0))
        example["candidate_metrics"] = selected.get("metrics", {})
    return example


def write_latents_bvh(agent, latents: torch.Tensor, output_file: Path) -> None:
    dconv = agent.posterior.decoder.decode_dynamic(latents)

    import VclSimuBackend

    character_to_bvh = VclSimuBackend.ODESim.CharacterTOBVH
    saver = character_to_bvh(agent.env.sim_character, 120)
    saver.bvh_hierarchy_no_root()
    observation, _ = agent.env.reset(0)

    for frame_idx in range(dconv.shape[1]):
        obs = observation["observation"]
        action, _ = agent.act_tracking(
            obs_history=[obs.reshape(1, 323)],
            target_latent=dconv[:, frame_idx],
        )
        action = ptu.to_numpy(action).flatten()
        for substep in range(6):
            saver.append_no_root_to_buffer()
            if substep == 0:
                step_generator = agent.env.step_core(action, using_yield=True)
            _ = next(step_generator)
        try:
            _ = next(step_generator)
        except StopIteration as exc:
            new_observation, _, _, _ = exc.value
        observation = new_observation

    output_file.parent.mkdir(parents=True, exist_ok=True)
    saver.to_file(str(output_file))


def measure_generated_latents(agent, latents: torch.Tensor, output_file: Path) -> dict[str, float]:
    from compute_bvh_proxy_metrics import compute_metrics

    write_latents_bvh(agent, latents, output_file)
    metrics = compute_metrics(output_file)
    return {key: float(value) for key, value in metrics.items() if isinstance(value, float)}


def build_cache(args: argparse.Namespace, device: torch.device):
    rows = read_style_rows(Path(args.train_csv), args.max_samples, args.seed)
    if not rows:
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

    motion_store = HumanMLMotionStore() if args.humanml_motion_selection else None
    selection_metrics = tuple(
        metric.strip()
        for metric in args.selection_metrics.split(",")
        if metric.strip()
    )
    candidate_bvh_dir = Path(args.candidate_bvh_dir) if args.candidate_bvh_dir else Path(args.output_dir) / "candidate_selection_bvh"

    examples = []
    for i, row in enumerate(rows, start=1):
        caption = row["caption"]
        target_metrics = None
        candidate_metric_fn = None
        if motion_store is not None and row.get("motion_id") and row.get("text_path"):
            try:
                target_metrics = motion_store.metrics_for(
                    Path(row["text_path"]),
                    row["motion_id"],
                    fps=args.humanml_fps,
                )
            except Exception as exc:
                print(f"warning: HumanML3D motion metrics unavailable for {row.get('motion_id')}: {exc}")

        if target_metrics is not None and args.teacher_samples > 1:
            def candidate_metric_fn(latents, sample_idx, *, row_index=i):
                output_file = candidate_bvh_dir / f"candidate_{row_index:05d}_{sample_idx:02d}.bvh"
                return measure_generated_latents(agent, latents, output_file)

        example = make_teacher_example(
            teacher,
            caption,
            tokenizer,
            encoder,
            device,
            args.max_length,
            teacher_samples=args.teacher_samples,
            target_metrics=target_metrics,
            candidate_metric_fn=candidate_metric_fn,
            selection_metrics=selection_metrics,
            teacher_categorical_sampling=args.teacher_categorical_sampling,
            teacher_top_k=args.teacher_top_k,
            teacher_temperature=args.teacher_temperature,
            sample_seed_base=args.seed * 100000 + i * 1000,
        )
        if example is not None:
            examples.append(example)
        detail = ""
        if example is not None and "selection_score" in example:
            detail = (
                f" selected={example['selected_sample_idx']}"
                f" score={float(example['selection_score']):.4f}"
            )
        print(f"cache {i}/{len(rows)} kept={len(examples)}{detail} caption={caption[:80]}")

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


def train_lora(args: argparse.Namespace, device: torch.device, examples) -> None:
    if not examples:
        raise RuntimeError("No cached examples available for training.")

    agent, _ = build_agent_clean(args.device)
    ptu.init_gpu(torch.cuda.is_available(), gpu_id=args.device)
    agent.simple_load(str(MOCONVQ / "moconvq_base.data"), strict=True)
    agent.eval()

    student = build_gpt(agent, device)
    load_pretrained_gpt(student, device)
    stats = inject_lora(
        student,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_prefixes=tuple(args.target_prefix.split(",")),
    )
    student.to(device)
    if args.resume_lora:
        checkpoint = torch.load(args.resume_lora, map_location="cpu")
        load_lora_state_dict(student, checkpoint["lora"])
    student.train()
    del agent
    clean_cuda()

    trainable = [param for param in student.parameters() if param.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method": "lora_distillation",
        "train_csv": args.train_csv,
        "cache_path": args.cache_path,
        "examples": len(examples),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "teacher_samples": args.teacher_samples,
        "teacher_categorical_sampling": args.teacher_categorical_sampling,
        "teacher_top_k": args.teacher_top_k,
        "teacher_temperature": args.teacher_temperature,
        "humanml_motion_selection": args.humanml_motion_selection,
        "humanml_fps": args.humanml_fps,
        "selection_metrics": args.selection_metrics,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_prefix": args.target_prefix,
        "lora_modules": stats.modules,
        "trainable_params": stats.trainable_params,
        "total_params": stats.total_params,
        "trainable_fraction": stats.trainable_params / max(stats.total_params, 1),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# LoRA Style Distillation",
                "",
                "This run freezes the pretrained MoConVQ text generator and trains only LoRA updates.",
                "Pseudo token targets are distilled from the pretrained teacher.",
                "When enabled, HumanML3D motion metrics select among multiple teacher samples before cache writing.",
                "",
                f"- Examples: {len(examples)}",
                f"- Teacher samples per caption: {args.teacher_samples}",
                f"- Teacher categorical sampling: {args.teacher_categorical_sampling}",
                f"- Teacher top-k: {args.teacher_top_k}",
                f"- Teacher temperature: {args.teacher_temperature}",
                f"- HumanML3D motion selection: {args.humanml_motion_selection}",
                f"- LoRA modules: {stats.modules}",
                f"- Trainable parameters: {stats.trainable_params}",
                f"- Total parameters: {stats.total_params}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(config, indent=2))

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
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
                    print(f"epoch={epoch} step={global_step} loss={loss_value:.6f}")

            ckpt_path = output_dir / f"style_lora_epoch{epoch}.pth"
            torch.save(
                {
                    "lora": lora_state_dict(student),
                    "config": config,
                    "epoch": epoch,
                    "step": global_step,
                },
                ckpt_path,
            )
            print(f"saved checkpoint: {ckpt_path}")

    torch.save(
        {"lora": lora_state_dict(student), "config": config, "step": global_step},
        output_dir / "style_lora_last.pth",
    )
    print(f"done: log={log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/style_subset_small/style_train_small.csv")
    parser.add_argument("--output-dir", default="outputs/finetune_lora")
    parser.add_argument("--cache-path", default="outputs/finetune_lora/distill_cache.pt")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--resume-lora", default=None)
    parser.add_argument("--teacher-samples", type=int, default=1)
    parser.add_argument("--teacher-categorical-sampling", action="store_true")
    parser.add_argument("--teacher-top-k", type=int, default=50)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--humanml-motion-selection", action="store_true")
    parser.add_argument("--humanml-fps", type=float, default=20.0)
    parser.add_argument("--selection-metrics", default=",".join(DEFAULT_SELECTION_METRICS))
    parser.add_argument("--candidate-bvh-dir", default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-prefix", default="trans_temporal")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if args.train_only:
        examples = torch.load(args.cache_path, map_location="cpu")
    else:
        examples = load_or_build_cache(args, device)
    if args.build_cache_only:
        return 0
    train_lora(args, device, examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
