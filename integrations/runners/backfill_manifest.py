"""
backfill_manifest.py

Enriches an EXISTING catalog with provenance that older runner versions
didn't store — full prompts, seed, parent_id — by reading them from the
seed-data files that produced each batch. No images are regenerated; this
only patches manifest.json and db/catalog.db in place.

Also supports rooting a batch's lineage at an origin entry: entries that
have no compare_to in their seed data get --default-parent (e.g. the
hand-uploaded image the whole style derives from).

Usage:
    python backfill_manifest.py \
        --catalog ../catalog_versioned_v3/catalog_versioned \
        --seed-data seed_data.json --default-parent 001 \
        --seed-data-2 seed_data_v2.json

    (repeat --seed-data-N for as many batch files as you have; any flag
     name starting with --seed-data works via nargs below)

Simpler: pass all seed files positionally:
    python backfill_manifest.py --catalog <path> --default-parent 001 \
        seed_data.json seed_data_v2.json seed_data_v3.json
"""

import argparse
import json
import sqlite3
from pathlib import Path


def ensure_db_columns(cur):
    existing = [row[1] for row in cur.execute("PRAGMA table_info(images)")]
    for col, coltype in (("batch", "TEXT"), ("workflow", "TEXT"),
                         ("seed", "INTEGER"), ("parent_id", "TEXT")):
        if col not in existing:
            cur.execute(f"ALTER TABLE images ADD COLUMN {col} {coltype}")
            print(f"  db: added '{col}' column")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_files", nargs="+", help="seed data json files, in batch order")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--default-parent", default=None,
                        help="parent_id for entries with no compare_to (e.g. the origin image id)")
    args = parser.parse_args()

    catalog_dir = Path(args.catalog).expanduser()
    manifest_path = catalog_dir / "manifest.json"
    db_path = catalog_dir / "db" / "catalog.db"

    manifest = json.loads(manifest_path.read_text())
    by_id = {e["id"]: e for e in manifest["entries"]}

    conn = sqlite3.connect(db_path) if db_path.exists() else None
    cur = conn.cursor() if conn else None
    if cur:
        ensure_db_columns(cur)

    patched = 0
    for seed_file in args.seed_files:
        seed_path = Path(seed_file).expanduser()
        seed = json.loads(seed_path.read_text())
        batch_name = seed.get("seed_batch_name", seed_path.stem)

        for entry in seed["entries"]:
            target = by_id.get(entry["id"])
            if target is None:
                print(f"  skip {entry['id']} ({entry['name']}) — not in catalog")
                continue

            prompts = {"positive": entry["positive_prompt"],
                       "negative": entry["negative_prompt"]}
            for extra in ("mask_prompt", "interior_prompt", "interior_negative"):
                if extra in entry:
                    prompts[extra] = entry[extra]
            target["prompts"] = prompts

            prov = target.setdefault("provenance", {})
            prov.setdefault("batch", batch_name)
            prov["seed"] = entry["seed"]
            parent = entry.get("compare_to") or args.default_parent
            # never let an entry parent itself
            prov["parent_id"] = parent if parent != entry["id"] else None
            target["tags"] = entry.get("tags", target.get("tags", []))

            if cur:
                cur.execute(
                    "UPDATE images SET seed = ?, parent_id = ?, batch = COALESCE(batch, ?) WHERE id = ?",
                    (entry["seed"], prov["parent_id"], batch_name, entry["id"]),
                )
            patched += 1
            print(f"  patched {entry['id']} ({entry['name']}): "
                  f"seed={entry['seed']} parent={prov['parent_id']} prompts={list(prompts)}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    if conn:
        conn.commit()
        conn.close()

    print(f"\nDone — {patched} entries enriched in {catalog_dir}")
    print("Tip: commit the catalog to record the backfill:  git add -A && git commit -m 'backfill provenance'")


if __name__ == "__main__":
    main()
