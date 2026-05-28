"""Generate BVH motions with a LoRA style-adapter checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

import torch


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

from style_lora import inject_lora, load_lora_state_dict  # noqa: E402


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


def strip_module_prefix(state_dict):
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def build_agent_clean(device_id: int):
    with moconvq_runtime():
        return build_agent(gpu=device_id)


def build_gpt(agent, device):
    embeddings = [
        torch.cat(
            [bottle_neck.embedding, torch.zeros_like(bottle_neck.embedding[:2])],
            dim=0,
        ).to(device)
        for bottle_neck in agent.posterior.bottle_neck_list
    ]
    return Text2Motion_Transformer(**GPT_CONFIG, embeddings=embeddings).to(device)


def load_pretrained_gpt(model, device):
    state = torch.load(MOCONVQ / "text_generation_GPT.pth", map_location=device)
    model.load_state_dict(strip_module_prefix(state), strict=True)


def read_prompts(args) -> list[tuple[int, str]]:
    if args.prompt_file:
        with Path(args.prompt_file).open("r", encoding="utf-8") as f:
            prompts = [
                (idx, line.strip())
                for idx, line in enumerate(f)
                if line.strip() and not line.lstrip().startswith("#")
            ]
    else:
        prompts = [(0, args.prompt)]
    if args.start_index is not None:
        prompts = [(idx, prompt) for idx, prompt in prompts if idx >= args.start_index]
    if args.end_index is not None:
        prompts = [(idx, prompt) for idx, prompt in prompts if idx <= args.end_index]
    return prompts


def text2bert(text: str, tokenizer, encoder, device):
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = encoder(**encoded)
    return output.last_hidden_state, ~encoded["attention_mask"].bool()


@torch.no_grad()
def generate_one(agent, gpt, text, output_file: Path, tokenizer, encoder, device, max_length: int):
    bert_feature, bert_mask = text2bert(text, tokenizer, encoder, device)
    clip_feature = torch.zeros((1, 512), device=device)
    cur_embedding, _ = gpt.sample(
        clip_feature,
        bert_feature,
        bert_mask,
        if_categorial=True,
        max_length=max_length,
    )
    dconv = agent.posterior.decoder.decode_dynamic(cur_embedding)

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/finetune_lora/style_lora_last.pth")
    parser.add_argument("--prompt", default="A person moves like doing tai chi.")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--output-dir", default="outputs/finetune_lora_samples")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        random_seed = int(args.seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})

    agent, _ = build_agent_clean(args.device)
    ptu.init_gpu(torch.cuda.is_available(), gpu_id=args.device)
    agent.simple_load(str(MOCONVQ / "moconvq_base.data"), strict=True)
    agent.eval()

    gpt = build_gpt(agent, device)
    load_pretrained_gpt(gpt, device)
    inject_lora(
        gpt,
        rank=int(config.get("lora_rank", 8)),
        alpha=float(config.get("lora_alpha", 16.0)),
        dropout=float(config.get("lora_dropout", 0.0)),
        target_prefixes=tuple(str(config.get("target_prefix", "trans_temporal")).split(",")),
    )
    gpt.to(device)
    load_lora_state_dict(gpt, checkpoint["lora"])
    gpt.eval()

    tokenizer = T5Tokenizer.from_pretrained("t5-large", resume_download=True)
    encoder = T5EncoderModel.from_pretrained("t5-large", resume_download=True).to(device)
    encoder.eval()

    prompts = read_prompts(args)
    out_dir = Path(args.output_dir)
    for ordinal, (idx, prompt) in enumerate(prompts, start=1):
        output_file = out_dir / f"evaluate_lora{idx}.bvh"
        print(f"[{ordinal}/{len(prompts)}] index={idx} {prompt}")
        generate_one(agent, gpt, prompt, output_file, tokenizer, encoder, device, args.max_length)
        print(f"wrote {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
