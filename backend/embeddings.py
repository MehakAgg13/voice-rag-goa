import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import create_chunks


DATA_PATH = "data/validation/hinval.parquet"
OUTPUT_DIR = "vector_db"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Final practical indexing size
MAX_RECORDS = 2000

# Maximum chunks stored per dataset record
MAX_CHUNKS_PER_RECORD = 10


def main():

    print("Loading dataset...")

    df = pd.read_parquet(DATA_PATH).head(MAX_RECORDS)

    print(f"Using {len(df)} records.")

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    all_chunks = []

    print("\nCreating controlled multi-strategy chunks...")

    for i, (_, row) in enumerate(df.iterrows()):

        chunks = create_chunks(row, model)

        # Keep a mixture of strategies
        selected = []

        for strategy in [
            "passage",
            "sentence",
            "fixed_overlap",
            "semantic"
        ]:

            strategy_chunks = [
                c for c in chunks
                if c["strategy"] == strategy
            ]

            # Take a small representative number
            selected.extend(strategy_chunks[:3])

        # Hard limit per record
        selected = selected[:MAX_CHUNKS_PER_RECORD]

        all_chunks.extend(selected)

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(df)} records")

    print(f"\nTotal chunks for indexing: {len(all_chunks)}")

    texts = [chunk["text"] for chunk in all_chunks]

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print("\nEmbeddings shape:", embeddings.shape)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "embeddings.npy"
        ),
        embeddings
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "chunks.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nSuccessfully saved:")

    print("vector_db/embeddings.npy")
    print("vector_db/chunks.json")

    print("\nIndex statistics:")

    strategies = {}

    for chunk in all_chunks:
        strategy = chunk["strategy"]
        strategies[strategy] = strategies.get(strategy, 0) + 1

    for strategy, count in strategies.items():
        print(f"{strategy}: {count}")


if __name__ == "__main__":
    main()