"""
embed_and_store.py

Run this on your RTX 3060 rig (not in the sandbox — needs a real GPU + your
local Python env). Generates CLIP image embeddings for every image in the
catalog and upserts them into a Qdrant collection, using the same metadata
already sitting in manifest.json.

Setup (one-time):
    pip install open_clip_torch torch qdrant-client pillow

Model choice for a 12GB 3060:
    ViT-L-14 (openai or laion2b_s32b_b82k weights) — ~1.7GB VRAM for inference,
    good embedding quality, plenty of headroom left for ComfyUI running
    alongside it. ViT-B-32 is faster/smaller if you want to batch thousands
    of images quickly and quality matters less.

Usage:
    python embed_and_store.py --catalog ./catalog --qdrant-url http://localhost:6333
"""

import argparse
import json
from pathlib import Path

import torch
import open_clip
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"  # swap to "laion2b_s32b_b82k" if you want the LAION weights
COLLECTION = "image_catalog"


def load_model(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    return model, preprocess


def embed_image(model, preprocess, image_path: Path, device: str):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features /= features.norm(dim=-1, keepdim=True)  # normalize for cosine similarity
    return features.squeeze(0).cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="./catalog", help="Path to catalog folder")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    catalog_dir = Path(args.catalog)
    manifest_path = catalog_dir / "manifest.json"
    images_dir = catalog_dir / "images"

    with open(manifest_path) as f:
        manifest = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = load_model(device)

    client = QdrantClient(url=args.qdrant_url)

    # embedding dim for ViT-L-14 is 768; change if you swap models
    embed_dim = 768
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION}'")

    points = []
    for entry in manifest["entries"]:
        image_path = images_dir / entry["filename"]
        if not image_path.exists():
            print(f"Skipping {entry['filename']} — not found")
            continue

        vector = embed_image(model, preprocess, image_path, device)
        points.append(
            PointStruct(
                id=int(entry["id"]),
                vector=vector,
                payload={
                    "filename": entry["filename"],
                    "name": entry["name"],
                    "category": entry["category"],
                    "tags": entry["tags"],
                    "description": entry["description"],
                },
            )
        )
        print(f"Embedded {entry['filename']} ({entry['name']})")

    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"Upserted {len(points)} point(s) into '{COLLECTION}'")


if __name__ == "__main__":
    main()
