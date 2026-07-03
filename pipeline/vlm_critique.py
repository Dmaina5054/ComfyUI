"""
vlm_critique.py

Generative critique stage — talks directly to llama.cpp's llama-server
(OpenAI-compatible /v1/chat/completions endpoint), NOT Ollama. Sends the
image + current tags/description and asks a vision-capable local model for
free-text critique and concrete prompt-rewrite suggestions for your next
ComfyUI/Flux generation.

Requires llama-server running with a vision (multimodal) GGUF model loaded
via --mmproj. Example launch on your rig (adjust paths/model to what you have):

    ./llama-server \
        -m models/Qwen2-VL-7B-Instruct-Q4_K_M.gguf \
        --mmproj models/qwen2-vl-7b-mmproj-f16.gguf \
        -fa on -ctk q4_0 -ctv q4_0 --context-shift -ngl 999 \
        --port 8080

Good vision model options for a 12GB 3060 (Q4_K_M quant, GGUF):
  - Qwen2-VL-7B-Instruct   — strong general vision-language, good detail recall
  - MiniCPM-V-2.6          — lighter, still solid, faster
  - LLaVA-1.6 (Mistral-7B) — older but well-tested with llama.cpp

Setup:
    pip install requests

Usage:
    python vlm_critique.py --catalog ../ --image brass_galaxy_globe_001.png \
        --endpoint http://localhost:8080/v1/chat/completions
"""

import argparse
import base64
import json
from pathlib import Path

import requests


SYSTEM_PROMPT = (
    "You are a visual critique assistant for an AI image-generation workflow "
    "(ComfyUI + Flux). Given an image and its current tags/description, give "
    "concise, concrete feedback: (1) what's working well, (2) 3-5 specific "
    "phrases that could be added to a future positive prompt to push the "
    "style further in a compelling direction, (3) one or two things to avoid "
    "repeating. Be specific and terse — no filler, no restating the obvious."
)


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_payload(image_b64: str, name: str, tags: list, description: str, model: str):
    user_text = (
        f"Image name: {name}\n"
        f"Current tags: {', '.join(tags)}\n"
        f"Current description: {description}\n\n"
        "Critique this image and suggest prompt improvements for the next "
        "generation in this style."
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.4,
        "max_tokens": 500,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="./catalog")
    parser.add_argument("--image", required=True, help="Filename inside catalog/images/")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8080/v1/chat/completions",
        help="Your llama-server OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--model",
        default="local-vlm",
        help="Model name field — llama-server generally ignores this and uses "
        "whatever's loaded, but the field is required by the API shape",
    )
    args = parser.parse_args()

    catalog_dir = Path(args.catalog)
    image_path = catalog_dir / "images" / args.image
    manifest_path = catalog_dir / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    entry = next((e for e in manifest["entries"] if e["filename"] == args.image), None)
    if entry is None:
        raise SystemExit(f"'{args.image}' not found in manifest.json")

    image_b64 = encode_image(image_path)
    payload = build_payload(
        image_b64, entry["name"], entry["tags"], entry["description"], args.model
    )

    print(f"Sending to {args.endpoint} ...")
    response = requests.post(args.endpoint, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    critique = result["choices"][0]["message"]["content"]
    print("\n--- VLM Critique ---\n")
    print(critique)

    # Save alongside the catalog entry so it's not lost
    out_path = catalog_dir / "critiques" / f"{Path(args.image).stem}_critique.txt"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(critique)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
