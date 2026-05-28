"""Run MoConVQ text-to-motion prompts through the local uv/venv Python."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moconvq-root", type=Path, default=Path("MoConVQ"))
    parser.add_argument("--prompt", default=None, help="single text prompt to generate")
    parser.add_argument("--prompt-file", type=Path, default=Path("data/prompts/baseline_and_style_prompts.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/project_prompts"))
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    python = args.python or (args.moconvq_root / ".venv/Scripts/python.exe")
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()

    cmd = [
        str(python),
        "Script/text2motion_generation.py",
    ]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    else:
        cmd += ["--prompt_file", str(args.prompt_file.resolve())]
    cmd += ["--output_dir", str(output_dir)]
    print(" ".join(cmd))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    return subprocess.call(cmd, cwd=args.moconvq_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
