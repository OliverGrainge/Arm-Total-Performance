#!/usr/bin/env python3
"""Download CIFAR-100 and generate CLIP embeddings for the image search tutorial.

This script:
  1. Downloads the CIFAR-100 dataset (~170 MB)
  2. Loads a CLIP ViT-L/14 model via open_clip
  3. Extracts 768-dimensional embeddings for all images
  4. Computes brute-force groundtruth for recall measurement
  5. Saves binary files for the C++ search programs and the dashboard

Requirements:
    pip install torch open-clip-torch Pillow numpy
"""

import argparse
import os
import pickle
import sys
import tarfile
import urllib.request

import numpy as np


def download_cifar100(data_dir):
    """Download and extract CIFAR-100 if not already present."""
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    archive_path = os.path.join(data_dir, "cifar-100-python.tar.gz")
    extract_dir = os.path.join(data_dir, "cifar-100-python")

    if os.path.exists(extract_dir):
        print("CIFAR-100 already downloaded.")
        return extract_dir

    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading CIFAR-100 from {url} ...")
    urllib.request.urlretrieve(url, archive_path)

    print("Extracting...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(data_dir)

    os.remove(archive_path)
    return extract_dir


def load_cifar100(extract_dir):
    """Load CIFAR-100 train and test splits."""
    with open(os.path.join(extract_dir, "train"), "rb") as f:
        train = pickle.load(f, encoding="bytes")
    with open(os.path.join(extract_dir, "test"), "rb") as f:
        test = pickle.load(f, encoding="bytes")

    # CIFAR-100 stores images as (N, 3072) uint8 in CHW order
    train_images = train[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_images = test[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.array(train[b"fine_labels"], dtype=np.int32)
    test_labels = np.array(test[b"fine_labels"], dtype=np.int32)

    meta_path = os.path.join(extract_dir, "meta")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    class_names = [name.decode("utf-8") for name in meta[b"fine_label_names"]]

    return train_images, train_labels, test_images, test_labels, class_names


def generate_clip_embeddings(images, model_name="ViT-B-32", batch_size=64):
    """Generate CLIP embeddings for an array of uint8 images."""
    import torch
    import open_clip
    from PIL import Image

    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai", device=device
    )
    model.eval()

    n = len(images)
    embeddings = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_pil = [Image.fromarray(img) for img in images[start:end]]
            batch_tensor = torch.stack(
                [preprocess(img) for img in batch_pil]
            ).to(device)
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())

            print(f"\r  Embedding: {end}/{n}", end="", flush=True)

    print()
    return np.vstack(embeddings).astype(np.float32)


def compute_groundtruth(base, queries, k):
    """Brute-force exact k-NN using L2 distance."""
    n_queries = queries.shape[0]
    gt = np.empty((n_queries, k), dtype=np.int32)
    batch = 256

    for start in range(0, n_queries, batch):
        end = min(start + batch, n_queries)
        q = queries[start:end]
        dists = (
            np.sum(q ** 2, axis=1, keepdims=True)
            + np.sum(base ** 2, axis=1, keepdims=False)
            - 2.0 * q @ base.T
        )
        gt[start:end] = np.argpartition(dists, k, axis=1)[:, :k]

        if (start // batch) % 5 == 0:
            print(f"\r  Groundtruth: {end}/{n_queries}", end="", flush=True)

    print()
    return gt


def main():
    parser = argparse.ArgumentParser(
        description="Download CIFAR-100 and generate CLIP embeddings"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Output directory (default: data)"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Top-K for groundtruth (default: 10)"
    )
    parser.add_argument(
        "--model", type=str, default="ViT-B-32",
        help="CLIP model name (default: ViT-B-32, alt: ViT-L-14)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="CLIP inference batch size (default: 64)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Download CIFAR-100 ──────────────────────────────────────
    extract_dir = download_cifar100(args.output_dir)

    # ── Step 2: Load images ─────────────────────────────────────────────
    print("Loading CIFAR-100 images...")
    train_imgs, train_labels, test_imgs, test_labels, class_names = (
        load_cifar100(extract_dir)
    )
    print(f"  Train: {len(train_imgs)} images  (database)")
    print(f"  Test:  {len(test_imgs)} images  (queries)")

    # ── Step 3: Generate CLIP embeddings ────────────────────────────────
    model_name = args.model
    print(f"\nGenerating CLIP {model_name} embeddings "
          "(this may take a few minutes on CPU)...\n")

    print("Database embeddings (train set):")
    base_emb = generate_clip_embeddings(
        train_imgs, model_name=model_name, batch_size=args.batch_size
    )

    print("Query embeddings (test set):")
    query_emb = generate_clip_embeddings(
        test_imgs, model_name=model_name, batch_size=args.batch_size
    )

    dim = base_emb.shape[1]
    n_base = base_emb.shape[0]
    n_query = query_emb.shape[0]

    # ── Step 4: Compute groundtruth ─────────────────────────────────────
    print(f"\nComputing exact top-{args.k} groundtruth (brute force)...")
    gt = compute_groundtruth(base_emb, query_emb, args.k)

    # ── Step 5: Save files ──────────────────────────────────────────────
    print("\nSaving files...")

    # Binary files consumed by the C++ search programs
    base_emb.tofile(os.path.join(args.output_dir, "embeddings.bin"))
    query_emb.tofile(os.path.join(args.output_dir, "queries.bin"))
    gt.tofile(os.path.join(args.output_dir, "groundtruth.bin"))

    # Numpy files consumed by the dashboard
    np.save(os.path.join(args.output_dir, "images.npy"), train_imgs)
    np.save(os.path.join(args.output_dir, "query_images.npy"), test_imgs)
    np.save(os.path.join(args.output_dir, "labels.npy"), train_labels)
    np.save(os.path.join(args.output_dir, "query_labels.npy"), test_labels)

    with open(os.path.join(args.output_dir, "class_names.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")

    # C++ header so the search programs know the dimensions
    header_path = os.path.join(args.output_dir, "data_config.h")
    with open(header_path, "w") as f:
        f.write("// Auto-generated by setup_data.py\n")
        f.write("#pragma once\n")
        f.write(f"#define DATA_N_BASE  {n_base}\n")
        f.write(f"#define DATA_N_QUERY {n_query}\n")
        f.write(f"#define DATA_DIM     {dim}\n")
        f.write(f"#define DATA_K       {args.k}\n")

    base_mb = os.path.getsize(
        os.path.join(args.output_dir, "embeddings.bin")
    ) / (1024 * 1024)
    query_mb = os.path.getsize(
        os.path.join(args.output_dir, "queries.bin")
    ) / (1024 * 1024)

    print(f"\nWritten to {args.output_dir}/:")
    print(f"  embeddings.bin     {n_base:>7,} x {dim}  ({base_mb:.1f} MB)")
    print(f"  queries.bin        {n_query:>7,} x {dim}  ({query_mb:.1f} MB)")
    print(f"  groundtruth.bin    {n_query:>7,} x {args.k}")
    print(f"  images.npy         {n_base:>7,} x 32 x 32 x 3")
    print(f"  query_images.npy   {n_query:>7,} x 32 x 32 x 3")
    print(f"  data_config.h      (C++ header)")
    print(f"\nEmbedding dimension: {dim} (CLIP {model_name})")
    print("Done!")


if __name__ == "__main__":
    main()
