"""
test_connection.py

Smoke test for the ComfyUI API pipeline — run this BEFORE generate_and_catalog.py
on the full seed batch. Checks each stage separately with clear pass/fail output:

  1. Is ComfyUI's API reachable at all?
  2. Are the required checkpoints (sd_xl_base_1.0 / sd_xl_refiner_1.0) actually
     visible to ComfyUI's object_info (catches path/naming mismatches before
     they surface as a cryptic queue error)?
  3. Queue ONE lightweight test job (entry 003, reduced steps for speed) and
     walk it through submit -> poll -> fetch -> save, printing progress at
     each stage.

Does NOT touch the catalog files — this is purely a connectivity/execution
test. Once this passes cleanly, generate_and_catalog.py is safe to run on
the full batch.

Usage:
    python test_connection.py --server 127.0.0.1:8188
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import requests


def check_server_reachable(server: str):
    print(f"[1/4] Checking ComfyUI is reachable at {server} ...")
    try:
        resp = requests.get(f"http://{server}/system_stats", timeout=10)
        resp.raise_for_status()
        stats = resp.json()
        vram = stats.get("devices", [{}])[0]
        print(f"      OK — {vram.get('name', 'GPU')}: "
              f"{vram.get('vram_free', '?')} / {vram.get('vram_total', '?')} bytes free/total")
        return True
    except requests.exceptions.ConnectionError:
        print(f"      FAILED — could not connect to {server}.")
        print("      Is ComfyUI running? Check the port matches your launch command.")
        return False
    except Exception as e:
        print(f"      FAILED — {e}")
        return False


def check_checkpoints_visible(server: str):
    print("[2/4] Checking required checkpoints are visible to ComfyUI ...")
    try:
        resp = requests.get(f"http://{server}/object_info/CheckpointLoaderSimple", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        available = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        required = ["sd_xl_base_1.0.safetensors", "sd_xl_refiner_1.0.safetensors"]
        all_ok = True
        for ckpt in required:
            if ckpt in available:
                print(f"      OK — {ckpt} found")
            else:
                print(f"      MISSING — {ckpt} not in ComfyUI's checkpoint list")
                all_ok = False
        if not all_ok:
            print(f"      ComfyUI currently sees: {available}")
        return all_ok
    except Exception as e:
        print(f"      FAILED — could not query object_info: {e}")
        return False


def run_test_generation(server: str):
    print("[3/4] Queuing a reduced-step test generation (entry 003, 8 steps instead of 25) ...")
    seed_dir = Path(__file__).parent
    template_path = seed_dir / "workflow_template.json"
    seed_data = json.loads((seed_dir / "seed_data.json").read_text())
    entry = next(e for e in seed_data["entries"] if e["id"] == "003")

    text = template_path.read_text()
    text = text.replace('"{{POSITIVE_PROMPT}}"', json.dumps(entry["positive_prompt"]))
    text = text.replace('"{{NEGATIVE_PROMPT}}"', json.dumps(entry["negative_prompt"]))
    text = text.replace('"{{SEED}}"', str(entry["seed"]))
    text = text.replace('"{{FILENAME_PREFIX}}"', json.dumps("connection_test"))
    workflow = json.loads(text)

    # speed up the smoke test — fewer steps, doesn't need to look good
    workflow["10"]["inputs"]["steps"] = 8
    workflow["10"]["inputs"]["end_at_step"] = 6
    workflow["11"]["inputs"]["steps"] = 8
    workflow["11"]["inputs"]["start_at_step"] = 6

    client_id = str(uuid.uuid4())
    try:
        resp = requests.post(
            f"http://{server}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        print(f"      FAILED — ComfyUI rejected the workflow: {resp.text}")
        return False
    except Exception as e:
        print(f"      FAILED — {e}")
        return False

    prompt_id = resp.json()["prompt_id"]
    print(f"      Queued OK — prompt_id: {prompt_id}")

    print("      Waiting for generation to finish (this uses the real GPU pipeline,")
    print("      may take 30s-2min depending on model swap time for base->refiner) ...")

    elapsed = 0
    timeout = 300
    while elapsed < timeout:
        resp = requests.get(f"http://{server}/history/{prompt_id}", timeout=15)
        history = resp.json()
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                print(f"      FAILED — generation errored: {status}")
                return False
            outputs = history[prompt_id]["outputs"]
            for node_output in outputs.values():
                if "images" in node_output:
                    img = node_output["images"][0]
                    print(f"      OK — image generated: {img['filename']}")
                    return fetch_and_save_test_image(server, img)
        time.sleep(3)
        elapsed += 3

    print(f"      FAILED — timed out after {timeout}s waiting for result")
    return False


def fetch_and_save_test_image(server: str, img: dict):
    print("[4/4] Fetching and saving the test image ...")
    try:
        resp = requests.get(
            f"http://{server}/view",
            params={"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]},
            timeout=30,
        )
        resp.raise_for_status()
        out_path = Path(__file__).parent / "connection_test_output.png"
        out_path.write_bytes(resp.content)
        print(f"      OK — saved to {out_path}")
        return True
    except Exception as e:
        print(f"      FAILED — {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1:8188")
    args = parser.parse_args()

    print("=" * 60)
    print("ComfyUI pipeline connection test")
    print("=" * 60)

    if not check_server_reachable(args.server):
        sys.exit(1)
    if not check_checkpoints_visible(args.server):
        print("\nStopping — fix checkpoint visibility before proceeding.")
        sys.exit(1)
    if not run_test_generation(args.server):
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — generate_and_catalog.py is safe to run on the full batch.")
    print("=" * 60)


if __name__ == "__main__":
    main()
