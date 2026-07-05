# CLAUDE.md — Catalog Factory

This repository contains **Catalog Factory** — a versioned AI image-generation
catalog system built as custom integrations on top of ComfyUI. This file is the
canonical context for Claude Code sessions in this tree. Repo:
`Dmaina5054/ComfyUI`, branch `local-minas`. Repo root is `integrations/`.
Repo layout note: integrations/ is a directory added inside a local clone of upstream ComfyUI (at ~/Projects/aifoss/comfyui/ on minas-tirith) — it is not part of ComfyUI itself. integrations/ has its own .git and is what's pushed to GitHub as Dmaina5054/ComfyUI (branch local-minas) — so despite the repo name, it contains only the integrations layer (runners, workflows, catalogs), not ComfyUI's source. The parent clone's .git belongs to upstream ComfyUI and is separate; integrations/ is excluded from it via .git/info/exclude. All paths in these instructions are relative to the integrations/ repo root.


## Operational rules (read first)


- **Never run generation batches without asking first.** Dry-run verification
  before live execution, always.
- **ComfyUI output folder:** `/home/frodo/Projects/aifoss/comfyui/output` —
  check here for `catalog_*_stage1` vs final image comparisons.
- `runners/` is the **only canonical tool location** — never trust or edit
  stale copies elsewhere in the tree.
- Verify actual file state on disk rather than assuming; this repo has been
  bitten by stale copies before.
- One experiment = one batch = **one variable changed**, git-tagged. Commit
  messages record the hypothesis and the findings.

## Architecture

- `runners/catalog_runner.py` submits seed batches
  (`catalogs/celestial_clockwork/pipeline/seed/seed_data*.json`) through
  parameterized workflow templates (`{{PLACEHOLDER}}` substitution — any entry
  field fills any matching template placeholder) to ComfyUI's `/prompt` API at
  `127.0.0.1:8188`, then registers each render into **five formats**:
  1. `manifest.csv`
  2. `manifest.json`
  3. `db/catalog.db` (SQLite: `images` + `tags` tables, auto-migrating schema)
  4. `training/` (Kohya image+caption pairs)
  5. `embedding_ready.json` (Qdrant stubs, vectors null until embedded)

  …with git auto-commit per batch recording batch name, seed file, workflow,
  and entry ids.
- `runners/backfill_manifest.py` retro-enriches provenance (prompts, seed,
  parent_id) from seed-data files; `--default-parent` roots lineage at an
  origin entry.
- `runners/test_connection.py` smoke-tests the ComfyUI API in 4 stages:
  reachability, checkpoint visibility, reduced-step generation, fetch.
- `browse.html` in the catalog root is the browsing UI — serve with
  `python3 -m http.server` from the catalog directory. Reads `manifest.json`;
  renders grid, search, batch filter, per-entry detail (seed, prompts,
  provenance), and clickable lineage chains.

## Conventions

- Entry ids are zero-padded strings (`"007"`), unique across the catalog, used
  as upsert identity in all five formats.
- Positive prompts **front-load subject + defining shape + must-have details
  in the first sentence, ≤90 tokens total** (SDXL CLIP dilutes content past
  ~75 tokens — proven by the v1→v2 ablation).
- Negative prompts name observed failure modes (e.g. `globe, sphere on a
  stand` to prevent bottle→globe drift; always include `UI elements, arrows,
  buttons, watermark, signature, cursive text`).
- `compare_to` records parentage. Lineage chains:
  - Bottle: 001 → 003 → 007 → 011
  - Terrarium: 001 → 004 → 008 → 012
  - Reliquary: 001 → 005 → 009 → 013
- Seeds are held identical across an entry's lineage so differences are
  attributable to the changed variable. **Reproducibility holds per-workflow
  only** — same seed + different graph = different image.
- Filenames encode entry id + slugified name; ComfyUI's `filename_prefix` is
  `catalog_{id}` (stage-1 saves get a `_stage1` suffix).

## Workflow versions (experiment ledger)

| Version | Tag | Approach | Result |
|---|---|---|---|
| v1 | `batch-001` | SDXL base+refiner, long prose prompts | Failed — subjects drifted (bottle→globe) |
| v2 | `batch-002-v2` | Same graph, front-loaded ≤90-token prompts | Subject identity recovered; interior detail still lost |
| v3 | `batch-003-v3` | Two-stage: SDXL base renders vessel with simple interior → CLIPSeg masks glass interior (`mask_prompt`) → juggernautXL_versionXInpaint repaints masked region at denoise 0.75 with dedicated `interior_prompt`. Saves both `_stage1` and final. Refiner dropped (3 checkpoints won't cycle through 12GB VRAM cleanly) | Under verification — see Open items |

v3 requires the **ComfyUI-CLIPSeg custom node** (installed; first run
downloads ~600MB segmentation weights).

## Hardware / stack

- Host **minas-tirith** — Debian 13 (trixie), kernel 6.12.86, x86_64,
  12 CPU threads.
- GPU: **RTX 3060 12GB**, driver 550.163.01, **CUDA 12.4** — pip installs
  should target **cu124** wheels.
- ComfyUI typically holds ~7GB VRAM resident (cached checkpoint) at idle —
  free the model cache or restart ComfyUI before co-running VRAM-hungry
  processes (e.g. a llama.cpp VLM).
- Checkpoints: sd_xl_base_1.0, sd_xl_refiner_1.0, juggernautXL_versionXInpaint,
  juggernautXL_ragnarokBy, realvisxlV50 Lightning, flux1DevFp8,
  ThinkDiffusionXL, architecturerealmix_v11, z-image-turbo-fp8, blindbox,
  512-inpainting-ema. Plus: 4x-UltraSharpV2 upscaler, ControlNet depth (SD15),
  depth_anything_v2_vitl, inspyrenet, CLIPSeg node.
- Also on host: **Qdrant** (`localhost:6333`); **llama.cpp server used
  directly — no Ollama** — flags:
  `-fa on -ctk q4_0 -ctv q4_0 --context-shift -ngl 999`.
- Python via venv; system-Python pip needs `--break-system-packages`.

## Downstream pipeline (built, partially run)

- `pipeline/embed_and_store.py` — CLIP ViT-L-14 image embeddings → Qdrant
  collection `image_catalog` (~2GB VRAM, coexists with ComfyUI).
- `pipeline/suggest_prompts.py` — scores a descriptor bank
  (lighting/material/mood/composition/style/texture) against an image via
  CLIP similarity; surfaces high-scoring unused terms as positive-prompt
  candidates.
- `pipeline/vlm_critique.py` — free-text critique via a vision GGUF
  (Qwen2-VL-7B / MiniCPM-V-2.6) served by llama-server with `--mmproj`,
  hitting `/v1/chat/completions`. Needs ComfyUI's VRAM freed first.

**Long-term goals:** grow catalogs to 30–50 consistent images for LoRA
training on the 3060 (`training/` pairs are already Kohya-format); apply the
same runner to a Kioo Labs real-estate staging catalog (architecturerealmix +
depth ControlNet); close the loop where critique output feeds the next seed
batch.

## Open items

1. **Verify catalog_011 stage1-vs-final** in the ComfyUI output folder — did
   the v3 inpaint actually add the orrery, or did the old runner register the
   stage1 image? (Runner is since fixed to prefer non-`_stage1` outputs.)
2. Assess 012/013 v3 results vs their v2 parents.
3. Run `embed_and_store.py` to populate real vectors in Qdrant.
4. Possible v4: tune CLIPSeg threshold (0.4→lower) / denoise (0.75→0.85), or
   add a mask-preview SaveImage node to make masks inspectable.
5. Entry 001's origin image must exist in the catalog `images/` for
   browse.html (copy `brass_galaxy_globe_001.png` if missing).

## Working style

Direct, no filler. Fix root causes, not symptoms. Test before shipping —
reproduce bugs, verify fixes against the failure case. One variable per
experiment. Flag honest limitations and sharp edges proactively. Verify
against the tree rather than assuming file state.