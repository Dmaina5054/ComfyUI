"""
suggest_prompts.py

The "suggest improvements" engine. Runs CLIP's text encoder against a bank
of candidate descriptors (style, lighting, material, mood, composition) and
scores each one against your image's embedding via cosine similarity. Prints
the highest-scoring descriptors you haven't already tagged — these are your
strongest candidates to add to a positive prompt for the *next* ComfyUI
generation in this style.

This is deliberately vocabulary-bank based (not a generative model) so it's
fast, cheap, and runs entirely on CLIP with no extra model download. If you
later want free-text generative critique ("this would look better with..."),
that needs a vision-language model (Qwen2-VL, LLaVA, moondream) served via
your existing llama-server / LM Studio setup — flag it and I'll build that
as a second stage (vlm_critique.py) that talks to your local OpenAI-compatible
endpoint.

Setup: same as embed_and_store.py (open_clip_torch, torch, pillow)

Usage:
    python suggest_prompts.py --catalog ./catalog --image brass_galaxy_globe_001.png
"""

import argparse
import json
from pathlib import Path

import torch
import open_clip
from PIL import Image


MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"

# Candidate descriptor bank, grouped by category. Extend this freely —
# it's the single lever that controls suggestion quality/breadth.
DESCRIPTOR_BANK = {
    "lighting": [
        "dramatic rim lighting", "soft diffused light", "volumetric god rays",
        "candlelight glow", "moody chiaroscuro", "bioluminescent glow",
        "golden hour lighting", "cold blue backlight", "harsh studio lighting",
    ],
    "material": [
        "aged brass", "polished copper", "tarnished silver", "weathered leather",
        "cracked porcelain", "frosted glass", "oxidized bronze", "dark oak wood",
        "wrought iron",
    ],
    "mood": [
        "melancholic atmosphere", "whimsical and playful", "ominous and foreboding",
        "serene and contemplative", "nostalgic", "otherworldly", "mysterious",
        "grand and majestic",
    ],
    "composition": [
        "rule of thirds framing", "extreme close-up macro detail", "wide establishing shot",
        "symmetrical composition", "shallow depth of field bokeh", "low angle heroic shot",
        "tilted dutch angle",
    ],
    "style": [
        "steampunk", "art nouveau", "gothic victorian", "solarpunk", "cyberpunk",
        "baroque ornamentation", "dieselpunk", "clockpunk", "cosmic horror",
    ],
    "texture_detail": [
        "intricate engraved detail", "hyper-realistic textures", "hand-painted miniature look",
        "photorealistic render", "fine particulate dust", "polished reflective surface",
    ],
}


def load_model(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    return model, preprocess, tokenizer


def score_descriptors(model, tokenizer, image_features, device: str):
    all_terms = [term for terms in DESCRIPTOR_BANK.values() for term in terms]
    text_tokens = tokenizer(all_terms).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0).cpu().tolist()

    return dict(zip(all_terms, similarities))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="./catalog")
    parser.add_argument("--image", required=True, help="Filename inside catalog/images/")
    parser.add_argument("--top-n", type=int, default=3, help="Top suggestions per category")
    args = parser.parse_args()

    catalog_dir = Path(args.catalog)
    image_path = catalog_dir / "images" / args.image
    manifest_path = catalog_dir / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    entry = next((e for e in manifest["entries"] if e["filename"] == args.image), None)
    existing_tags = set(t.lower() for t in entry["tags"]) if entry else set()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess, tokenizer = load_model(device)

    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    scores = score_descriptors(model, tokenizer, image_features, device)

    print(f"\nSuggested additions for '{args.image}' (already-tagged terms are skipped):\n")
    for category, terms in DESCRIPTOR_BANK.items():
        ranked = sorted(
            [(t, scores[t]) for t in terms if t.lower() not in existing_tags],
            key=lambda x: x[1],
            reverse=True,
        )[: args.top_n]
        print(f"  {category}:")
        for term, score in ranked:
            print(f"    {term}  (similarity: {score:.3f})")
        print()


if __name__ == "__main__":
    main()
