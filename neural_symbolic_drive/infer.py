"""Run Neural-Symbolic Drive inference on ShareGPT-style JSON data."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def build_messages(item: dict) -> list[dict]:
    """Convert one ShareGPT-style item into a Hugging Face multimodal message."""
    user_msg = item["messages"][0]
    images = item.get("images", [])
    text = user_msg["content"]
    parts = text.split("<image>")
    if len(parts) - 1 != len(images):
        raise ValueError(
            f"image count mismatch: prompt has {len(parts) - 1} <image> tags, "
            f"but item has {len(images)} image paths"
        )

    content = []
    for i, chunk in enumerate(parts):
        if chunk:
            content.append({"type": "text", "text": chunk})
        if i < len(images):
            content.append({"type": "image", "image": images[i]})
    return [{"role": "user", "content": content}]


def load_images(paths: list[str], base_dir: Path) -> list[Image.Image]:
    images = []
    for path in paths:
        image_path = Path(path)
        if not image_path.is_absolute():
            image_path = base_dir / image_path
        image = Image.open(image_path)
        image.load()
        if image.mode != "RGB":
            image = image.convert("RGB")
        images.append(image)
    return images


def get_label(item: dict) -> str:
    for message in reversed(item.get("messages", [])):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="xiangbog/Neural-Symbolic-Drive",
        help="Local checkpoint path or Hugging Face repo id",
    )
    parser.add_argument("--data_json", required=True, help="ShareGPT-style JSON list")
    parser.add_argument("--save_path", required=True, help="Output JSONL path")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_pixels", type=int, default=262144)
    parser.add_argument("--device_map", default="auto", help='Passed to from_pretrained, e.g. "cuda" or "auto"')
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_total", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if args.shard_total > 1:
        save_path = save_path.with_suffix(
            f".shard{args.shard_idx}of{args.shard_total}{save_path.suffix}"
        )

    print(f"Loading processor and model from {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    data_path = Path(args.data_json)
    data = json.loads(data_path.read_text())
    image_base_dir = data_path.resolve().parent
    if args.limit:
        data = data[: args.limit]

    indices = list(range(args.shard_idx, len(data), args.shard_total))
    shard_data = [(i, data[i]) for i in indices]
    print(f"Running {len(shard_data)} / {len(data)} samples", flush=True)

    start = time.time()
    with save_path.open("w", encoding="utf-8") as fout:
        for orig_idx, item in tqdm(shard_data):
            images = []
            try:
                messages = build_messages(item)
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                images = load_images(item.get("images", []), image_base_dir)
                inputs = processor(
                    text=[prompt],
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)

                for key, value in list(inputs.items()):
                    if torch.is_tensor(value) and value.dtype == torch.float32:
                        inputs[key] = value.to(torch.bfloat16)

                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )

                new_tokens = generated[:, inputs["input_ids"].shape[1] :]
                prediction = processor.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                row = {
                    "orig_idx": orig_idx,
                    "id": item.get("id", str(orig_idx)),
                    "predict": prediction,
                    "label": get_label(item),
                }
            except Exception as exc:
                row = {
                    "orig_idx": orig_idx,
                    "id": item.get("id", str(orig_idx)),
                    "predict": "",
                    "label": get_label(item),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                for image in images:
                    image.close()

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            if torch.cuda.is_available() and orig_idx % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    print(f"Wrote {save_path} in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
