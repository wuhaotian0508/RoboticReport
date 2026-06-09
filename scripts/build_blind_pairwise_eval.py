"""Build a blind baseline-vs-LoRA pairwise human evaluation page."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path


LABEL_FIELDS = [
    "item_id",
    "evaluator",
    "semantic_a_1_to_5",
    "semantic_b_1_to_5",
    "style_a_1_to_5",
    "style_b_1_to_5",
    "plausibility_a_1_to_5",
    "plausibility_b_1_to_5",
    "preference_a_b_tie",
    "notes",
]


def read_prompts(path: Path, max_items: int) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return prompts[:max_items] if max_items else prompts


def resolve_bvh(directory: Path, prefix: str, idx: int) -> Path:
    candidates = [
        directory / f"{prefix}{idx}.bvh",
        directory / "project_prompts" / f"{prefix}{idx}.bvh",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def render_preview(bvh_path: Path, output_path: Path, title: str, fps: int, max_frames: int) -> Path:
    if output_path.exists():
        return output_path
    fallback = output_path.with_suffix(".gif")
    if fallback.exists():
        return fallback

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "render_bvh_video.py"),
        str(bvh_path),
        "--output",
        str(output_path),
        "--title",
        title,
        "--fps",
        str(fps),
        "--max-frames",
        str(max_frames),
    ]
    subprocess.run(cmd, check=True)
    if output_path.exists():
        return output_path
    if fallback.exists():
        return fallback
    raise RuntimeError(f"Renderer did not create {output_path} or {fallback}")


def copy_blind_media(source: Path, target: Path) -> Path:
    target = target.with_suffix(source.suffix.lower())
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    return target


def relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def media_html(path: str, label: str) -> str:
    escaped_path = html.escape(path)
    escaped_label = html.escape(label)
    suffix = Path(path).suffix.lower()
    if suffix == ".gif":
        return f'<img class="preview" src="{escaped_path}" alt="{escaped_label} preview">'
    return f'<video controls preload="metadata" src="{escaped_path}"></video>'


def write_labels_template(items: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "item_id": item["item_id"],
                    "evaluator": "",
                    "semantic_a_1_to_5": "",
                    "semantic_b_1_to_5": "",
                    "style_a_1_to_5": "",
                    "style_b_1_to_5": "",
                    "plausibility_a_1_to_5": "",
                    "plausibility_b_1_to_5": "",
                    "preference_a_b_tie": "",
                    "notes": "",
                }
            )


def build_html(items: list[dict[str, str]], output: Path) -> str:
    data = [
        {
            "item_id": item["item_id"],
            "prompt": item["prompt"],
        }
        for item in items
    ]
    sections = []
    for item in items:
        item_id = html.escape(item["item_id"])
        prompt = html.escape(item["prompt"])
        a_media = media_html(item["a_media"], "A")
        b_media = media_html(item["b_media"], "B")
        sections.append(
            f"""
<section class="item" data-item-id="{item_id}">
  <h2>Item {item_id}</h2>
  <p class="prompt">{prompt}</p>
  <div class="grid">
    <article>
      <h3>A</h3>
      {a_media}
      <div class="scores">
        <label>Semantic <input type="number" min="1" max="5" data-field="semantic_a_1_to_5"></label>
        <label>Style <input type="number" min="1" max="5" data-field="style_a_1_to_5"></label>
        <label>Plausibility <input type="number" min="1" max="5" data-field="plausibility_a_1_to_5"></label>
      </div>
    </article>
    <article>
      <h3>B</h3>
      {b_media}
      <div class="scores">
        <label>Semantic <input type="number" min="1" max="5" data-field="semantic_b_1_to_5"></label>
        <label>Style <input type="number" min="1" max="5" data-field="style_b_1_to_5"></label>
        <label>Plausibility <input type="number" min="1" max="5" data-field="plausibility_b_1_to_5"></label>
      </div>
    </article>
  </div>
  <div class="preference">
    <label><input type="radio" name="pref_{item_id}" value="A"> Prefer A</label>
    <label><input type="radio" name="pref_{item_id}" value="B"> Prefer B</label>
    <label><input type="radio" name="pref_{item_id}" value="Tie"> Tie</label>
  </div>
  <textarea data-field="notes" placeholder="notes"></textarea>
</section>
"""
        )

    payload = json.dumps(data)
    fields = json.dumps(LABEL_FIELDS)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind Pairwise Motion Evaluation</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8fafc; color: #111827; }}
.toolbar {{ position: sticky; top: 0; background: #f8fafc; padding: 12px 0; z-index: 2; }}
button {{ border: 1px solid #111827; background: #111827; color: #fff; padding: 8px 12px; cursor: pointer; }}
.item {{ border-top: 1px solid #d1d5db; padding: 24px 0; }}
.prompt {{ font-weight: 700; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
article {{ background: white; border: 1px solid #d1d5db; padding: 12px; }}
video, .preview {{ width: 100%; aspect-ratio: 1 / 1; background: #111827; object-fit: contain; }}
.scores {{ display: grid; gap: 8px; margin-top: 10px; }}
input[type="number"] {{ width: 64px; }}
.preference {{ display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }}
textarea {{ box-sizing: border-box; width: 100%; min-height: 64px; margin-top: 12px; }}
pre {{ white-space: pre-wrap; background: white; border: 1px solid #d1d5db; padding: 12px; }}
</style>
</head>
<body>
<h1>Blind Pairwise Motion Evaluation</h1>
<p>Rate A and B against the prompt, then choose A, B, or Tie. Model identities are hidden.</p>
<label>Evaluator <input id="evaluator" placeholder="name or id"></label>
<div class="toolbar"><button type="button" onclick="exportCsv()">Export preference_labels.csv</button></div>
{''.join(sections)}
<h2>CSV Preview</h2>
<pre id="preview"></pre>
<script>
const items = {payload};
const fields = {fields};
function csvEscape(value) {{
  const text = String(value ?? "");
  if (/[",\\n]/.test(text)) return '"' + text.replaceAll('"', '""') + '"';
  return text;
}}
function valueFor(section, field) {{
  const node = section.querySelector(`[data-field="${{field}}"]`);
  return node ? node.value : "";
}}
function buildRows() {{
  const rows = [fields];
  const evaluator = document.getElementById("evaluator").value;
  for (const item of items) {{
    const section = document.querySelector(`section[data-item-id="${{item.item_id}}"]`);
    const selected = section.querySelector(`input[name="pref_${{item.item_id}}"]:checked`);
    rows.push([
      item.item_id,
      evaluator,
      valueFor(section, "semantic_a_1_to_5"),
      valueFor(section, "semantic_b_1_to_5"),
      valueFor(section, "style_a_1_to_5"),
      valueFor(section, "style_b_1_to_5"),
      valueFor(section, "plausibility_a_1_to_5"),
      valueFor(section, "plausibility_b_1_to_5"),
      selected ? selected.value : "",
      valueFor(section, "notes"),
    ]);
  }}
  return rows;
}}
function buildCsv() {{
  return buildRows().map((row) => row.map(csvEscape).join(",")).join("\\n") + "\\n";
}}
function refreshPreview() {{
  document.getElementById("preview").textContent = buildCsv();
}}
function exportCsv() {{
  const blob = new Blob([buildCsv()], {{ type: "text/csv;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "preference_labels.csv";
  link.click();
  URL.revokeObjectURL(url);
}}
document.addEventListener("change", refreshPreview);
document.addEventListener("input", refreshPreview);
refreshPreview();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=Path, default=Path("outputs/human_pairwise_eval/test_prompts.txt"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/human_pairwise_eval/baseline_bvh"))
    parser.add_argument("--lora-dir", type=Path, default=Path("outputs/human_pairwise_eval/lora_bvh"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/human_pairwise_eval"))
    parser.add_argument("--max-items", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-fps", type=int, default=20)
    parser.add_argument("--render-max-frames", type=int, default=180)
    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file, args.max_items)
    rng = random.Random(args.seed)
    media_dir = args.output_dir / "media"
    key_rows = []
    review_items = []

    for idx, prompt in enumerate(prompts):
        baseline_bvh = resolve_bvh(args.baseline_dir, "evaluate_gpt", idx)
        lora_bvh = resolve_bvh(args.lora_dir, "evaluate_lora", idx)
        missing = [str(path) for path in (baseline_bvh, lora_bvh) if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing BVH pair for item " + str(idx) + ": " + ", ".join(missing))

        baseline_preview = render_preview(
            baseline_bvh,
            args.output_dir / "rendered" / f"baseline_{idx:03d}.mp4",
            f"item_{idx:03d}_source_1",
            args.render_fps,
            args.render_max_frames,
        )
        lora_preview = render_preview(
            lora_bvh,
            args.output_dir / "rendered" / f"lora_{idx:03d}.mp4",
            f"item_{idx:03d}_source_2",
            args.render_fps,
            args.render_max_frames,
        )
        item_id = f"item_{idx:03d}"
        sides = [("baseline", baseline_preview), ("lora", lora_preview)]
        rng.shuffle(sides)
        a_model, a_source = sides[0]
        b_model, b_source = sides[1]
        a_media = copy_blind_media(a_source, media_dir / f"{item_id}_A")
        b_media = copy_blind_media(b_source, media_dir / f"{item_id}_B")

        key_rows.append(
            {
                "item_id": item_id,
                "prompt_index": idx,
                "prompt": prompt,
                "a_model": a_model,
                "b_model": b_model,
                "baseline_bvh": str(baseline_bvh),
                "lora_bvh": str(lora_bvh),
                "a_media": str(a_media),
                "b_media": str(b_media),
            }
        )
        review_items.append(
            {
                "item_id": item_id,
                "prompt": prompt,
                "a_media": relative(a_media, args.output_dir),
                "b_media": relative(b_media, args.output_dir),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    key_path = args.output_dir / "blind_key.csv"
    with key_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    labels_template = args.output_dir / "labels_template.csv"
    write_labels_template(review_items, labels_template)
    review_html = args.output_dir / "review.html"
    review_html.write_text(build_html(review_items, review_html), encoding="utf-8")
    print(f"wrote {key_path}")
    print(f"wrote {labels_template}")
    print(f"wrote {review_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
