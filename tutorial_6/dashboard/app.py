#!/usr/bin/env python3
"""Text-to-Image Search Dashboard powered by pgvector.

Type a text description and find matching images from a database of 50,000
CIFAR-100 photographs. Uses CLIP to encode text queries into the same
embedding space as the images, then searches with pgvector on PostgreSQL.

Run from the tutorial_6 directory:

    python dashboard/app.py

Requirements:
    pip install -r requirements.txt
    PostgreSQL with pgvector must be running (see scripts/setup_data.py)
"""

import os
import time

import numpy as np
import torch
import open_clip
import gradio as gr
import psycopg2
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_NAME = os.environ.get("PGDATABASE", "clip_search")

# ── Load CLIP text encoder ────────────────────────────────────────────
print("Loading CLIP text encoder...")
clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device="cpu"
)
clip_model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# ── Load image data for display ───────────────────────────────────────
print("Loading image data...")
images = np.load(os.path.join(DATA_DIR, "images.npy"))
labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
with open(os.path.join(DATA_DIR, "class_names.txt")) as f:
    class_names = [line.strip() for line in f]

n_images = len(images)
print(f"Ready — {n_images:,} images in database")


def search(query_text):
    """Encode text query with CLIP and search pgvector."""
    if not query_text or not query_text.strip():
        return [], ""

    # Encode text → 512-dim embedding
    with torch.no_grad():
        tokens = tokenizer([query_text])
        text_emb = clip_model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb.cpu().numpy().flatten().astype(np.float32)

    # Query pgvector
    emb_str = "[" + ",".join(f"{x:.6f}" for x in text_emb) + "]"

    conn = psycopg2.connect(dbname=DB_NAME)
    cur = conn.cursor()
    cur.execute("SET hnsw.ef_search = 100")

    t0 = time.perf_counter()
    cur.execute(
        "SELECT id, label, embedding <-> %s::vector AS distance "
        "FROM images ORDER BY embedding <-> %s::vector LIMIT 10",
        (emb_str, emb_str)
    )
    results = cur.fetchall()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    cur.close()
    conn.close()

    # Build gallery
    gallery = []
    for img_id, label, dist in results:
        img = Image.fromarray(images[img_id - 1]).resize((128, 128), Image.NEAREST)
        gallery.append((img, f"{label}  (L2: {dist:.4f})"))

    stats = f"Found {len(results)} results in **{elapsed_ms:.2f} ms**"
    return gallery, stats


# ── Gradio UI ─────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_color="#334155",
    block_border_color_dark="#334155",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_title_text_color="#f1f5f9",
    block_title_text_color_dark="#f1f5f9",
    input_background_fill="#1e293b",
    input_background_fill_dark="#1e293b",
    input_border_color="#475569",
    input_border_color_dark="#475569",
    input_placeholder_color="#64748b",
    input_placeholder_color_dark="#64748b",
    button_primary_background_fill="#3b82f6",
    button_primary_background_fill_dark="#3b82f6",
    button_primary_background_fill_hover="#2563eb",
    button_primary_background_fill_hover_dark="#2563eb",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    border_color_primary="#334155",
    border_color_primary_dark="#334155",
)

css = """
.gradio-container { max-width: 900px !important; }
.gallery-item img { border-radius: 6px; }
footer { display: none !important; }
"""

with gr.Blocks(theme=theme, css=css, title="CLIP Image Search") as demo:
    gr.Markdown(
        f"# CLIP Text-to-Image Search\n"
        f"Search **{n_images:,}** CIFAR-100 images by text description. "
        f"Images are indexed as **512**-dimensional CLIP embeddings "
        f"in [pgvector](https://github.com/pgvector/pgvector) on PostgreSQL."
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="Describe an image",
            placeholder="e.g. a red sports car, a cute puppy, sunset over the ocean...",
            scale=4,
        )
        search_btn = gr.Button("Search", variant="primary", scale=1)

    stats_output = gr.Markdown("")
    gallery_output = gr.Gallery(
        label="Results",
        columns=5,
        height="auto",
        object_fit="contain",
    )

    gr.Examples(
        examples=[
            ["a red sports car"],
            ["sunset over the ocean"],
            ["a cute puppy"],
            ["forest with tall trees"],
            ["a plate of food"],
            ["a big orange fish"],
        ],
        inputs=query_input,
    )

    search_btn.click(
        fn=search, inputs=query_input, outputs=[gallery_output, stats_output]
    )
    query_input.submit(
        fn=search, inputs=query_input, outputs=[gallery_output, stats_output]
    )

if __name__ == "__main__":
    # Monkey-patch gradio_client bug where additionalProperties=True (a bool)
    # gets passed to get_type() which tries "const" in schema on it.
    import gradio_client.utils as _gc_utils
    _orig_get_type = _gc_utils.get_type
    def _patched_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)
    _gc_utils.get_type = _patched_get_type

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
