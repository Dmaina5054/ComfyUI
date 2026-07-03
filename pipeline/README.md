# Local CLIP Pipeline — run on the RTX 3060 rig

This sandbox can't run these (no GPU, restricted network) — copy the
`pipeline/` folder to your machine and run it there alongside ComfyUI.

## 1. Setup (one-time)

```bash
cd catalog/pipeline
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`open_clip_torch` will pull the ViT-L-14 (openai) weights on first run
(~900MB download, cached afterward). VRAM footprint is small — roughly
1.5–2GB — so it comfortably coexists with ComfyUI/Flux on the 3060's 12GB.

## 2. Generate embeddings + push to Qdrant

```bash
python embed_and_store.py --catalog ../  --qdrant-url http://localhost:6333
```

This reads `manifest.json`, embeds every listed image, and upserts vectors
+ metadata (name, tags, description) into a Qdrant collection called
`image_catalog`. Re-run any time you add new images to the catalog —
it upserts by id, so existing entries just get refreshed.

Once this is populated you can do real similarity search — "find catalog
images closest to this new one" — using standard Qdrant queries.

## 3. Get prompt-improvement suggestions

```bash
python suggest_prompts.py --catalog ../ --image brass_galaxy_globe_001.png
```

This scores the image against a bank of descriptor candidates (lighting,
material, mood, composition, style, texture) and prints the top-scoring
ones per category that aren't already in the image's tags. These are your
strongest candidates to fold into the *next* ComfyUI positive prompt when
generating something in the same vein — e.g. it might surface
`"volumetric god rays"` or `"dramatic rim lighting"` as high-scoring but
unused, meaning the model already "sees" that quality in the image even
though it's not written down anywhere yet.

Edit `DESCRIPTOR_BANK` in `suggest_prompts.py` to expand the vocabulary —
that's the main lever for suggestion quality. Worth seeding it with terms
pulled from your best past ComfyUI/Flux prompts.

## 4. Generative critique (llama.cpp server — no Ollama)

`vlm_critique.py` gets you the free-text critique stage: "this composition
would read stronger from a lower angle," etc. It talks directly to
llama-server's OpenAI-compatible `/v1/chat/completions` endpoint — nothing
routes through Ollama anywhere in this pipeline.

Requires llama-server running with a vision-capable GGUF model loaded via
`--mmproj`. Example, using the same flags you already run:

```bash
./llama-server \
    -m models/Qwen2-VL-7B-Instruct-Q4_K_M.gguf \
    --mmproj models/qwen2-vl-7b-mmproj-f16.gguf \
    -fa on -ctk q4_0 -ctv q4_0 --context-shift -ngl 999 \
    --port 8080
```

Good vision model options for the 3060 (Q4_K_M GGUF quant):
- **Qwen2-VL-7B-Instruct** — strong detail recall, good default choice
- **MiniCPM-V-2.6** — lighter/faster if VRAM is tight alongside Flux
- **LLaVA-1.6 (Mistral-7B)** — older, but well-proven with llama.cpp

Run it:

```bash
pip install requests
python vlm_critique.py --catalog ../ --image brass_galaxy_globe_001.png \
    --endpoint http://localhost:8080/v1/chat/completions
```

It sends the image + current tags/description, asks for what's working,
3-5 concrete phrases to add to the next positive prompt, and what to avoid
repeating — then saves the response to `catalog/critiques/`.
