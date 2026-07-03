# Seed Batch — celestial_clockwork_batch_001

This is the initial seed data for the catalog: one parameterized ComfyUI
workflow + 5 prompt variants, run automatically and registered into every
catalog format we've built so far.

## Files
- `workflow_template.json` — your uploaded SDXL base+refiner workflow, with
  the positive prompt, negative prompt, seed, and filename prefix swapped
  for placeholders (`{{POSITIVE_PROMPT}}`, `{{NEGATIVE_PROMPT}}`, `{{SEED}}`,
  `{{FILENAME_PREFIX}}`). Structure and node graph are otherwise untouched.
  Used only by `generate_and_catalog.py` — never opened directly in ComfyUI.
- `workflows_gui/` — the same 5 workflows, but concrete: placeholders
  filled in, no template syntax, ready to drag-and-drop load in the ComfyUI
  GUI. Each file is named `{catalog_id}_{catalog_name_slug}.json` so the
  filename on disk always matches the corresponding catalog entry:
  - `002_celestial_potion_bottle_raw.json`
  - `003_celestial_potion_bottle.json`
  - `004_bioluminescent_clockwork_terrarium.json`
  - `005_solar_clockwork_reliquary.json`
  - `006_clockwork_nebula_macro_detail.json`

  If you want these browsable/loadable inside the ComfyUI app itself, copy
  this folder's contents into ComfyUI's workflow directory (recent
  versions: `ComfyUI/user/default/workflows/`, older: `ComfyUI/workflows/`).
  Not required for `generate_and_catalog.py` to work — that talks to
  ComfyUI purely over its `/prompt` HTTP API and never touches these files.
- `seed_data.json` — 5 entries:
  - **002** — your original prompt, byte-for-byte, kept unedited. It contains
    leftover UI-artifact phrasing ("navigation arrows," "X and Run buttons,"
    "full-screen view") that risks rendering literally in the image. Kept as
    a reference/comparison point, tagged `raw-prompt-has-ui-artifacts`.
  - **003** — the same concept, cleaned: UI-artifact language removed, and a
    negative-prompt guard added (`UI elements, navigation arrows, buttons,
    on-screen overlays`) so SDXL doesn't try to render them.
  - **004–006** — new compositions in the same steampunk/celestial vocabulary
    (terrarium, warm-tone reliquary, macro close-up), meant to diversify the
    catalog rather than repeat one image five times.
- `generate_and_catalog.py` — submits each entry to your local ComfyUI,
  waits for the result, and writes it into `manifest.csv`, `manifest.json`,
  `db/catalog.db`, `training/`, and `embedding_ready.json` all at once.

## Run it

Needs ComfyUI running locally with `sd_xl_base_1.0.safetensors` and
`sd_xl_refiner_1.0.safetensors` loaded (matching your uploaded workflow),
API reachable at `127.0.0.1:8188` (ComfyUI's default).

```bash
cd catalog/pipeline/seed
pip install requests
python generate_and_catalog.py --catalog ../.. --server 127.0.0.1:8188
```

This queues all 5 jobs sequentially (base+refiner each, ~25 steps), waits
for each to finish, then writes the catalog entries. On a 3060 expect each
image to take a couple minutes.

## After it finishes

Run the rest of the pipeline against the freshly grown catalog:

```bash
cd ..
python embed_and_store.py --catalog ../ --qdrant-url http://localhost:6333
python suggest_prompts.py --catalog ../ --image 003_celestial_potion_bottle.png
```

## Growing the batch further

Add more entries to `seed_data.json` (same shape: id, name, category, tags,
seed, positive_prompt, negative_prompt) and re-run
`generate_and_catalog.py` — it's idempotent per id, so re-running with the
same id just refreshes that entry rather than duplicating it.
