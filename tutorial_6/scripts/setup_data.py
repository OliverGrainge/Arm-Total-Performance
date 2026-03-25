#!/usr/bin/env python3
"""Download CIFAR-100, generate CLIP embeddings, and load into PostgreSQL with pgvector.

This script:
  1. Downloads the CIFAR-100 dataset (~170 MB)
  2. Loads a CLIP ViT-B-32 model and extracts 512-dimensional embeddings
  3. Creates a PostgreSQL database and loads the embeddings into pgvector
  4. Saves image arrays and metadata for the dashboard

Requirements:
    pip install -r requirements.txt
    sudo apt install postgresql postgresql-16-pgvector   # on Ubuntu/Graviton
"""

import argparse
import os
import pickle
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
        tar.extractall(data_dir, filter="data")

    os.remove(archive_path)
    return extract_dir


def load_cifar100(extract_dir):
    """Load CIFAR-100 train and test splits."""
    with open(os.path.join(extract_dir, "train"), "rb") as f:
        train = pickle.load(f, encoding="bytes")
    with open(os.path.join(extract_dir, "test"), "rb") as f:
        test = pickle.load(f, encoding="bytes")

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

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai", device="cpu"
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
            )
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy())
            print(f"\r  {end}/{n}", end="", flush=True)

    print()
    return np.vstack(embeddings).astype(np.float32)


def load_into_pgvector(embeddings, labels, class_names, db_name):
    """Create a PostgreSQL database and load embeddings with pgvector."""
    import psycopg2

    # Connect to default database to create ours
    conn = psycopg2.connect(dbname="postgres")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
    cur.execute(f"CREATE DATABASE {db_name}")
    cur.close()
    conn.close()

    # Connect to the new database
    conn = psycopg2.connect(dbname=db_name)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    dim = embeddings.shape[1]

    cur.execute(f"""
        CREATE TABLE images (
            id SERIAL PRIMARY KEY,
            label TEXT NOT NULL,
            embedding vector({dim})
        )
    """)

    # Bulk insert in batches
    print("Loading embeddings into PostgreSQL...")
    batch_size = 500
    n = len(embeddings)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        values = []
        for i in range(start, end):
            label = class_names[labels[i]]
            emb_str = "[" + ",".join(f"{x:.6f}" for x in embeddings[i]) + "]"
            values.append(cur.mogrify("(%s, %s::vector)", (label, emb_str)).decode())
        cur.execute(
            "INSERT INTO images (label, embedding) VALUES " + ",".join(values)
        )
        conn.commit()
        print(f"\r  {end}/{n}", end="", flush=True)
    print()

    # Build HNSW index
    print("Building HNSW index (this may take a few minutes)...")
    cur.execute(f"""
        CREATE INDEX ON images
        USING hnsw (embedding vector_l2_ops)
        WITH (m = 16, ef_construction = 200)
    """)
    conn.commit()

    # Verify
    cur.execute("SELECT COUNT(*) FROM images")
    count = cur.fetchone()[0]
    print(f"Loaded {count} vectors into pgvector database '{db_name}'")

    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Download CIFAR-100, generate CLIP embeddings, load into pgvector"
    )
    parser.add_argument("--data-dir", default="data", help="Output directory (default: data)")
    parser.add_argument("--db-name", default="clip_search", help="PostgreSQL database name")
    parser.add_argument("--model", default="ViT-B-32", help="CLIP model name")
    parser.add_argument("--batch-size", type=int, default=64, help="CLIP inference batch size")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Step 1: Download CIFAR-100
    extract_dir = download_cifar100(args.data_dir)

    # Step 2: Load images
    print("Loading CIFAR-100 images...")
    train_imgs, train_labels, test_imgs, test_labels, class_names = (
        load_cifar100(extract_dir)
    )
    print(f"  Train: {len(train_imgs)} images  (database)")
    print(f"  Test:  {len(test_imgs)} images  (queries)")

    # Step 3: Generate CLIP embeddings (or load if already saved)
    emb_path = os.path.join(args.data_dir, "embeddings.npy")
    query_emb_path = os.path.join(args.data_dir, "query_embeddings.npy")

    if os.path.exists(emb_path) and os.path.exists(query_emb_path):
        print("Embeddings already generated, loading from disk...")
        base_emb = np.load(emb_path)
        query_emb = np.load(query_emb_path)
    else:
        print(f"\nGenerating CLIP {args.model} embeddings (may take a few minutes on CPU)...\n")
        print("Database embeddings (train set):")
        base_emb = generate_clip_embeddings(
            train_imgs, model_name=args.model, batch_size=args.batch_size
        )
        print("Query embeddings (test set):")
        query_emb = generate_clip_embeddings(
            test_imgs, model_name=args.model, batch_size=args.batch_size
        )
        np.save(emb_path, base_emb)
        np.save(query_emb_path, query_emb)

    dim = base_emb.shape[1]
    print(f"  Database: {base_emb.shape} ({base_emb.dtype})")
    print(f"  Queries:  {query_emb.shape} ({query_emb.dtype})")

    # Step 4: Save images and labels for dashboard
    np.save(os.path.join(args.data_dir, "images.npy"), train_imgs)
    np.save(os.path.join(args.data_dir, "query_images.npy"), test_imgs)
    np.save(os.path.join(args.data_dir, "labels.npy"), train_labels)
    np.save(os.path.join(args.data_dir, "query_labels.npy"), test_labels)
    with open(os.path.join(args.data_dir, "class_names.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")

    # Step 5: Save query embeddings as binary for benchmark
    query_emb.tofile(os.path.join(args.data_dir, "queries.bin"))

    # Step 6: Load into pgvector
    load_into_pgvector(base_emb, train_labels, class_names, db_name=args.db_name)

    print(f"\nSetup complete!")
    print(f"  Database:  {args.db_name}")
    print(f"  Vectors:   {base_emb.shape[0]} images, {query_emb.shape[0]} queries")
    print(f"  Dimension: {dim}")


if __name__ == "__main__":
    main()
