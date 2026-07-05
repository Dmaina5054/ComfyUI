"""
catalog_runner.py

Standalone, catalog-agnostic generation runner. Lives WHEREVER you keep your
tools — completely decoupled from any catalog. One installed copy can
populate any number of independent, separately-versioned catalogs.

Everything is a parameter:
    --workflow    path to a workflow template (with {{PLACEHOLDERS}})
    --seed-data   path to a seed batch json
    --catalog     path to the catalog root to populate (created if missing)
    --server      ComfyUI API host:port
    --git-commit  after a successful run, git-add + commit the catalog
                  (init's the repo if the catalog isn't one yet)

Examples — one tool, many catalogs:

    # steampunk catalog, v2 batch
    python catalog_runner.py \
        --workflow ./templates/sdxl_base_refiner.json \
        --seed-data ./batches/celestial_v2.json \
        --catalog ~/catalogs/steampunk \
        --git-commit

    # completely separate real-estate staging catalog, same tool
    python catalog_runner.py \
        --workflow ./templates/architecturerealmix_depth.json \
        --seed-data ./batches/staging_batch_001.json \
        --catalog ~/catalogs/kioo_staging \
        --git-commit

Each catalog gets its own git history; the auto-commit message records the
batch name, seed-data file, workflow template, and entry ids — so every
image in every catalog is traceable to exactly what produced it.

Setup: pip install requests
"""

import argparse
import csv
import json
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

import requests


# ---------------------------------------------------------------- workflow

def load_workflow(template_path: Path, entry: dict) -> dict:
    """Generic placeholder substitution: every '{{KEY}}' in the template is
    filled from the entry field of the same (lowercased) name. Strings are
    JSON-escaped; numbers are inserted bare. FILENAME_PREFIX is derived from
    the entry id. Fails loudly if the template still contains unfilled
    placeholders afterward (catches template/seed-data mismatches)."""
    text = template_path.read_text()
    # whole-value form first: "{{KEY}}" -> JSON-escaped value / bare number
    text = text.replace('"{{FILENAME_PREFIX}}"', json.dumps(f"catalog_{entry['id']}"))
    for key, value in entry.items():
        placeholder = f'"{{{{{key.upper()}}}}}"'
        if isinstance(value, str):
            text = text.replace(placeholder, json.dumps(value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            text = text.replace(placeholder, str(value))
    # in-string form second: {{KEY}} inside a larger string (e.g. "{{FILENAME_PREFIX}}_stage1")
    text = text.replace('{{FILENAME_PREFIX}}', f"catalog_{entry['id']}")
    for key, value in entry.items():
        if isinstance(value, str):
            # escape for embedding inside an existing JSON string
            embedded = json.dumps(value)[1:-1]
            text = text.replace(f'{{{{{key.upper()}}}}}', embedded)
    import re as _re
    leftover = _re.findall(r'\{\{[A-Z_]+\}\}', text)
    if leftover:
        raise SystemExit(
            f"Template {template_path.name} has unfilled placeholders "
            f"{sorted(set(leftover))} — entry '{entry.get('id')}' is missing "
            f"those fields in the seed data."
        )
    return json.loads(text)


# ---------------------------------------------------------------- comfyui api

def queue_prompt(server: str, workflow: dict, client_id: str) -> str:
    resp = requests.post(
        f"http://{server}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def find_target_node_id(workflow: dict, entry_id: str) -> str:
    """The 'real' final output is the SaveImage node whose filename_prefix
    is the bare 'catalog_{id}' with no suffix — stage1/_mask/etc variants
    always have a suffix appended. Resolved by contract, not by execution
    order (which is unreliable — see wait_for_result)."""
    expected_prefix = f"catalog_{entry_id}"
    matches = [
        node_id for node_id, node in workflow.items()
        if isinstance(node, dict)
        and node.get("class_type") == "SaveImage"
        and node.get("inputs", {}).get("filename_prefix") == expected_prefix
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one SaveImage node with filename_prefix "
            f"'{expected_prefix}' (no suffix) for entry '{entry_id}', "
            f"found {len(matches)}: {matches}. Check the workflow template "
            f"for missing/duplicate bare-prefix SaveImage nodes."
        )
    return matches[0]


def wait_for_result(server: str, prompt_id: str, target_node_id: str,
                    poll_interval=2, timeout=600):
    elapsed = 0
    while elapsed < timeout:
        resp = requests.get(f"http://{server}/history/{prompt_id}", timeout=30)
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"Generation errored: {status}")
            outputs = history[prompt_id]["outputs"]
            if target_node_id in outputs and "images" in outputs[target_node_id]:
                img = outputs[target_node_id]["images"][0]
                return img["filename"], img["subfolder"], img["type"]
            # else: execution still in progress or target node hasn't
            # produced output yet — keep polling until timeout.
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(
        f"Timed out waiting for node {target_node_id}'s output on prompt {prompt_id}"
    )


def fetch_image(server: str, filename: str, subfolder: str, image_type: str) -> bytes:
    resp = requests.get(
        f"http://{server}/view",
        params={"filename": filename, "subfolder": subfolder, "type": image_type},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------- catalog write

def register_in_catalog(catalog_dir: Path, entry: dict, image_bytes: bytes,
                        batch_name: str, workflow_name: str):
    catalog_dir.mkdir(parents=True, exist_ok=True)

    image_filename = f"{entry['id']}_" + "".join(
        c if c.isalnum() or c == "_" else "_" for c in entry["name"].lower().replace(" ", "_")
    ).strip("_") + ".png"

    # 1. image
    images_dir = catalog_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / image_filename).write_bytes(image_bytes)

    description = entry["positive_prompt"][:400]
    today = time.strftime("%Y-%m-%d")

    # 2. manifest.csv
    csv_path = catalog_dir / "manifest.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "filename", "name", "category", "tags",
                             "description", "date_added", "source", "batch", "workflow"])
        writer.writerow([entry["id"], image_filename, entry["name"], entry["category"],
                         ",".join(entry["tags"]), description, today,
                         "comfyui", batch_name, workflow_name])

    # 3. manifest.json
    json_path = catalog_dir / "manifest.json"
    manifest = json.loads(json_path.read_text()) if json_path.exists() else {
        "catalog_name": catalog_dir.name, "version": "1.0", "entries": []
    }
    manifest["entries"] = [e for e in manifest["entries"] if e["id"] != entry["id"]]
    prompts = {"positive": entry["positive_prompt"], "negative": entry["negative_prompt"]}
    for extra in ("mask_prompt", "interior_prompt", "interior_negative"):
        if extra in entry:
            prompts[extra] = entry[extra]
    manifest["entries"].append({
        "id": entry["id"],
        "filename": image_filename,
        "name": entry["name"],
        "category": entry["category"],
        "tags": entry["tags"],
        "description": description,
        "prompts": prompts,
        "provenance": {
            "date_added": today,
            "source": "comfyui",
            "seed": entry["seed"],
            "batch": batch_name,
            "workflow": workflow_name,
            "parent_id": entry.get("compare_to"),
        },
        "training_ready": True,
        "embedding": None,
    })
    json_path.write_text(json.dumps(manifest, indent=2))

    # 4. sqlite
    db_path = catalog_dir / "db" / "catalog.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS images (
        id TEXT PRIMARY KEY, filename TEXT, name TEXT, category TEXT,
        description TEXT, date_added TEXT, source TEXT, batch TEXT, workflow TEXT,
        seed INTEGER, parent_id TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS tags (image_id TEXT, tag TEXT)""")
    # migrate catalogs created by older runner versions
    existing_cols = [row[1] for row in cur.execute("PRAGMA table_info(images)")]
    for col, coltype in (("batch", "TEXT"), ("workflow", "TEXT"),
                         ("seed", "INTEGER"), ("parent_id", "TEXT")):
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE images ADD COLUMN {col} {coltype}")
            print(f"  Migrated db schema: added '{col}' column")
    cur.execute("INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry["id"], image_filename, entry["name"], entry["category"],
                 description, today, "comfyui", batch_name, workflow_name,
                 entry["seed"], entry.get("compare_to")))
    cur.execute("DELETE FROM tags WHERE image_id = ?", (entry["id"],))
    for tag in entry["tags"]:
        cur.execute("INSERT INTO tags VALUES (?, ?)", (entry["id"], tag))
    conn.commit()
    conn.close()

    # 5. training pair
    training_dir = catalog_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_filename).stem
    (training_dir / f"{stem}.png").write_bytes(image_bytes)
    (training_dir / f"{stem}.txt").write_text(", ".join(entry["tags"]))

    # 6. embedding stub
    emb_path = catalog_dir / "embedding_ready.json"
    emb = json.loads(emb_path.read_text()) if emb_path.exists() else {
        "collection": catalog_dir.name, "points": []
    }
    emb["points"] = [p for p in emb["points"] if p["id"] != entry["id"]]
    emb["points"].append({
        "id": entry["id"],
        "payload": {"filename": image_filename, "name": entry["name"],
                    "category": entry["category"], "tags": entry["tags"],
                    "description": description},
        "vector": None,
        "vector_model_suggested": "CLIP ViT-L/14 (image embedding)",
    })
    emb_path.write_text(json.dumps(emb, indent=2))

    print(f"  Registered '{entry['name']}' -> {image_filename}")


# ---------------------------------------------------------------- git

DEFAULT_GITIGNORE = """images/
training/*.png
__pycache__/
*.pyc
"""


def git(catalog_dir: Path, *args, check=True):
    return subprocess.run(["git", "-C", str(catalog_dir), *args],
                          capture_output=True, text=True, check=check)


def git_commit_catalog(catalog_dir: Path, batch_name: str, seed_data_path: Path,
                       workflow_path: Path, entry_ids: list):
    if not (catalog_dir / ".git").exists():
        git(catalog_dir, "init")
        gitignore = catalog_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(DEFAULT_GITIGNORE)
        print(f"  Initialized new git repo in {catalog_dir}")

    git(catalog_dir, "add", "-A")
    status = git(catalog_dir, "status", "--porcelain").stdout.strip()
    if not status:
        print("  Nothing to commit (no changes).")
        return

    msg = (f"batch: {batch_name}\n\n"
           f"seed-data: {seed_data_path.name}\n"
           f"workflow: {workflow_path.name}\n"
           f"entries: {', '.join(entry_ids)}\n"
           f"date: {time.strftime('%Y-%m-%d %H:%M')}")
    git(catalog_dir, "-c", "user.name=catalog-runner",
        "-c", "user.email=catalog-runner@local", "commit", "-m", msg)
    short = git(catalog_dir, "rev-parse", "--short", "HEAD").stdout.strip()
    print(f"  Committed as {short}: batch {batch_name}")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Populate any versioned catalog from any workflow+seed batch")
    parser.add_argument("--workflow", required=True, help="Path to workflow template json (with {{PLACEHOLDERS}})")
    parser.add_argument("--seed-data", required=True, help="Path to seed batch json")
    parser.add_argument("--catalog", required=True, help="Path to catalog root to populate (created if missing)")
    parser.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI host:port")
    parser.add_argument("--git-commit", action="store_true",
                        help="After the run, git add+commit the catalog (init repo if needed)")
    args = parser.parse_args()

    workflow_path = Path(args.workflow).expanduser()
    seed_data_path = Path(args.seed_data).expanduser()
    catalog_dir = Path(args.catalog).expanduser()

    for p, label in [(workflow_path, "--workflow"), (seed_data_path, "--seed-data")]:
        if not p.exists():
            raise SystemExit(f"{label} file not found: {p}")

    seed_data = json.loads(seed_data_path.read_text())
    batch_name = seed_data.get("seed_batch_name", seed_data_path.stem)
    client_id = str(uuid.uuid4())
    done_ids = []

    print(f"Batch: {batch_name} ({len(seed_data['entries'])} entries)")
    print(f"Catalog: {catalog_dir}")
    print(f"Workflow: {workflow_path.name}\n")

    for entry in seed_data["entries"]:
        print(f"Generating: {entry['name']} (id {entry['id']}, seed {entry['seed']})")
        workflow = load_workflow(workflow_path, entry)
        target_node_id = find_target_node_id(workflow, entry["id"])
        prompt_id = queue_prompt(args.server, workflow, client_id)
        filename, subfolder, image_type = wait_for_result(args.server, prompt_id, target_node_id)
        image_bytes = fetch_image(args.server, filename, subfolder, image_type)
        register_in_catalog(catalog_dir, entry, image_bytes, batch_name, workflow_path.name)
        done_ids.append(entry["id"])

    if args.git_commit:
        print("\nCommitting catalog state:")
        git_commit_catalog(catalog_dir, batch_name, seed_data_path, workflow_path, done_ids)

    print(f"\nDone — {len(done_ids)} entries registered in {catalog_dir}")


if __name__ == "__main__":
    main()