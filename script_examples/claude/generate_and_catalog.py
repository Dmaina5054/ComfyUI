"""
generate_and_catalog.py

Runs on your machine (needs ComfyUI running locally, default API at
http://127.0.0.1:8188). For each entry in seed_data.json:
  1. Fills workflow_template.json with that entry's prompt/negative/seed
  2. Queues it via ComfyUI's /prompt API
  3. Polls until the image is ready, downloads it
  4. Registers the result into ALL catalog formats we already built:
     manifest.csv, manifest.json, db/catalog.db, training/ pair,
     embedding_ready.json (vector left null — run embed_and_store.py after)

Setup:
    pip install requests

Usage:
    python generate_and_catalog.py --catalog ../.. --server 127.0.0.1:8188
    (run from catalog/pipeline/seed/, or adjust --catalog path)
"""

import argparse
import csv
import json
import sqlite3
import time
import uuid
from pathlib import Path

import requests


def load_workflow(template_path: Path, entry: dict) -> dict:
    text = template_path.read_text()
    text = text.replace('"{{POSITIVE_PROMPT}}"', json.dumps(entry["positive_prompt"]))
    text = text.replace('"{{NEGATIVE_PROMPT}}"', json.dumps(entry["negative_prompt"]))
    text = text.replace('"{{SEED}}"', str(entry["seed"]))
    prefix = f"catalog_{entry['id']}"
    text = text.replace('"{{FILENAME_PREFIX}}"', json.dumps(prefix))
    return json.loads(text)


def queue_prompt(server: str, workflow: dict, client_id: str) -> str:
    resp = requests.post(
        f"http://{server}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_result(server: str, prompt_id: str, poll_interval=2, timeout=600):
    elapsed = 0
    while elapsed < timeout:
        resp = requests.get(f"http://{server}/history/{prompt_id}", timeout=30)
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history:
            outputs = history[prompt_id]["outputs"]
            for node_output in outputs.values():
                if "images" in node_output:
                    img = node_output["images"][0]
                    return img["filename"], img["subfolder"], img["type"]
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def fetch_image(server: str, filename: str, subfolder: str, image_type: str) -> bytes:
    resp = requests.get(
        f"http://{server}/view",
        params={"filename": filename, "subfolder": subfolder, "type": image_type},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def register_in_catalog(catalog_dir: Path, entry: dict, image_bytes: bytes):
    image_filename = f"{entry['id']}_{entry['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"

    # 1. save image
    images_dir = catalog_dir / "images"
    images_dir.mkdir(exist_ok=True)
    image_path = images_dir / image_filename
    image_path.write_bytes(image_bytes)

    description = entry["positive_prompt"][:400]  # trimmed for catalog description field

    # 2. manifest.csv
    csv_path = catalog_dir / "manifest.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "filename", "name", "category", "tags", "description", "date_added", "source"])
        writer.writerow([
            entry["id"], image_filename, entry["name"], entry["category"],
            ",".join(entry["tags"]), description, time.strftime("%Y-%m-%d"), "comfyui-seed-batch"
        ])

    # 3. manifest.json
    json_path = catalog_dir / "manifest.json"
    manifest = json.loads(json_path.read_text()) if json_path.exists() else {
        "catalog_name": "image_catalog", "version": "1.0", "entries": []
    }
    manifest["entries"] = [e for e in manifest["entries"] if e["id"] != entry["id"]]  # replace if re-run
    manifest["entries"].append({
        "id": entry["id"],
        "filename": image_filename,
        "name": entry["name"],
        "category": entry["category"],
        "tags": entry["tags"],
        "description": description,
        "provenance": {
            "date_added": time.strftime("%Y-%m-%d"),
            "source": "comfyui-seed-batch",
            "seed": entry["seed"],
        },
        "training_ready": True,
        "embedding": None,
    })
    json_path.write_text(json.dumps(manifest, indent=2))

    # 4. sqlite db
    db_path = catalog_dir / "db" / "catalog.db"
    db_path.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS images (
        id TEXT PRIMARY KEY, filename TEXT, name TEXT, category TEXT,
        description TEXT, date_added TEXT, source TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS tags (image_id TEXT, tag TEXT)""")
    cur.execute("INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?)", (
        entry["id"], image_filename, entry["name"], entry["category"],
        description, time.strftime("%Y-%m-%d"), "comfyui-seed-batch"
    ))
    cur.execute("DELETE FROM tags WHERE image_id = ?", (entry["id"],))
    for tag in entry["tags"]:
        cur.execute("INSERT INTO tags VALUES (?, ?)", (entry["id"], tag))
    conn.commit()
    conn.close()

    # 5. training pair
    training_dir = catalog_dir / "training"
    training_dir.mkdir(exist_ok=True)
    stem = Path(image_filename).stem
    (training_dir / f"{stem}.png").write_bytes(image_bytes)
    (training_dir / f"{stem}.txt").write_text(", ".join(entry["tags"]))

    # 6. embedding_ready.json
    emb_path = catalog_dir / "embedding_ready.json"
    emb = json.loads(emb_path.read_text()) if emb_path.exists() else {
        "collection": "image_catalog", "points": []
    }
    emb["points"] = [p for p in emb["points"] if p["id"] != entry["id"]]
    emb["points"].append({
        "id": entry["id"],
        "payload": {
            "filename": image_filename, "name": entry["name"], "category": entry["category"],
            "tags": entry["tags"], "description": description,
        },
        "vector": None,
        "vector_model_suggested": "CLIP ViT-L/14 (image embedding)",
    })
    emb_path.write_text(json.dumps(emb, indent=2))

    print(f"  Registered '{entry['name']}' -> {image_filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="../..", help="Path to catalog/ root")
    parser.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI server host:port")
    parser.add_argument("--seed-data", default="seed_data.json")
    parser.add_argument("--workflow", default="workflow_template.json")
    args = parser.parse_args()

    catalog_dir = Path(args.catalog)
    seed_data = json.loads(Path(args.seed_data).read_text())
    template_path = Path(args.workflow)
    client_id = str(uuid.uuid4())

    print(f"Seed batch: {seed_data['seed_batch_name']} ({len(seed_data['entries'])} entries)")
    for entry in seed_data["entries"]:
        print(f"\nGenerating: {entry['name']} (id {entry['id']}, seed {entry['seed']})")
        workflow = load_workflow(template_path, entry)
        prompt_id = queue_prompt(args.server, workflow, client_id)
        filename, subfolder, image_type = wait_for_result(args.server, prompt_id)
        image_bytes = fetch_image(args.server, filename, subfolder, image_type)
        register_in_catalog(catalog_dir, entry, image_bytes)

    print("\nDone. Run embed_and_store.py next to generate CLIP embeddings for the new entries.")


if __name__ == "__main__":
    main()