# Image Catalog — Multi-Format System

This catalog stores the same information in four different shapes, each suited
to a different computing use case. All four currently hold one entry:
**Steampunk Celestial Orrery** (`brass_galaxy_globe_001.png`).

## 1. `manifest.csv`
Flat, spreadsheet-readable format. Best for: quick browsing, importing into
Excel/Google Sheets, or any tool that just wants a simple table.

## 2. `manifest.json`
Structured, nested format. Best for: feeding into a script, an API, or a web
app — anything that wants typed fields (arrays for tags, nested objects for
attributes/provenance) rather than flat columns.

## 3. `db/catalog.db` (SQLite)
Real relational database with two tables (`images`, `tags`, joined by
`image_id`). Best for: querying at scale later — e.g. "give me every image
tagged `steampunk`" — once you have hundreds/thousands of entries. Query it
with any SQLite client or `sqlite3 catalog.db "SELECT * FROM images;"`.

## 4. `training/` (image + caption pairs)
`brass_galaxy_globe_001.png` + `brass_galaxy_globe_001.txt`. This is the
standard format used by diffusion/LoRA training tools (Kohya, ComfyUI
training scripts, etc.) — one image, one same-named `.txt` file with the
caption/tags. Best for: if you ever fine-tune a LoRA on your RTX 3060 rig
using this and future images.

## 5. `embedding_ready.json`
Schema shaped for a vector database (matches what your Qdrant setup expects).
The `vector` field is `null` here since generating a real embedding needs a
model (e.g. CLIP) — your LM Studio rig or the Qdrant RAG pipeline you've
already built can populate this. Best for: similarity search / "find images
like this one" / RAG over your image collection.

## Adding the next image
When you add image #002, update all four in parallel:
1. Add a row to `manifest.csv`
2. Add an entry to `manifest.json` → `entries[]`
3. Insert a row into `catalog.db` (`images` + `tags` tables)
4. Drop the image + matching `.txt` caption into `training/`
5. Add a point to `embedding_ready.json` → `points[]`

Once you're doing this for more than a handful of images, it's worth writing
a small script that takes one input (image + tags + description) and writes
out all five automatically, so they never drift out of sync.
