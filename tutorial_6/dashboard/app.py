#!/usr/bin/env python3
"""Text-to-Image Search Dashboard — type a description, find matching images.

Uses CLIP to encode text queries into the same embedding space as the
CIFAR-100 image database, then searches with HNSWlib.

Run from the tutorial_6 directory:

    python dashboard/app.py

Requirements:
    pip install gradio hnswlib numpy Pillow torch open-clip-torch
"""

import os
import time

import numpy as np
import torch
import open_clip
import gradio as gr
from PIL import Image

# ── Data directory (relative to tutorial_6/) ────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_data():
    """Load images, embeddings, CLIP text encoder, and build HNSW index."""
    import hnswlib

    print("Loading data...")
    base_images = np.load(os.path.join(DATA_DIR, "images.npy"))
    n_base = base_images.shape[0]

    base_embeddings = np.fromfile(
        os.path.join(DATA_DIR, "embeddings.bin"), dtype=np.float32
    )
    dim = len(base_embeddings) // n_base
    base_embeddings = base_embeddings.reshape(n_base, dim)

    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))

    with open(os.path.join(DATA_DIR, "class_names.txt")) as f:
        class_names = [line.strip() for line in f]

    # Load CLIP text encoder
    print("Loading CLIP text encoder...")
    model_name = "ViT-B-32"
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained="openai", device="cpu"
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Build HNSW index
    print(f"Building HNSW index ({n_base} vectors, dim={dim})...")
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n_base, ef_construction=200, M=32)
    index.add_items(base_embeddings, np.arange(n_base))
    index.set_ef(200)
    print("Ready!")

    return base_images, labels, class_names, clip_model, tokenizer, index


base_images, labels, class_names, clip_model, clip_tokenizer, index = load_data()


def search(query_text):
    """Encode text query with CLIP and search the HNSW index."""
    if not query_text or not query_text.strip():
        return [], ""

    tokens = clip_tokenizer([query_text])
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    query_vec = text_features.cpu().numpy().astype(np.float32)

    t0 = time.perf_counter()
    result_ids, distances = index.knn_query(query_vec, k=10)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = []
    for rid, dist in zip(result_ids[0], distances[0]):
        img = Image.fromarray(base_images[rid]).resize((128, 128), Image.NEAREST)
        caption = f"{class_names[labels[rid]]}  (L2: {dist:.4f})"
        results.append((img, caption))

    stats = f"Found {len(results)} results in **{elapsed_ms:.2f} ms**"
    return results, stats


# ── Gradio UI ───────────────────────────────────────────────────────────

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

n_base = len(base_images)
dim = index.dim

with gr.Blocks(theme=theme, css=css, title="CLIP Image Search") as demo:
    gr.Markdown(
        f"# CLIP Text-to-Image Search\n"
        f"Search **{n_base:,}** CIFAR-100 images by text description. "
        f"Images are indexed as **{dim}**-dimensional CLIP embeddings "
        f"in an HNSWlib graph — the same vector search that the C++ "
        f"tutorial optimises with ATP."
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
    demo.launch()
