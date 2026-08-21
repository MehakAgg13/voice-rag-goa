import os
import json
import time

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = "vector_db/embeddings.npy"
CHUNKS_PATH = "vector_db/chunks.json"
INDEX_PATH = "vector_db/faiss.index"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_index():

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH)

    print("Embedding shape:", embeddings.shape)

    # Inner Product + normalized embeddings = cosine similarity
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    print("Adding embeddings to FAISS...")

    index.add(embeddings)

    print("FAISS index size:", index.ntotal)

    faiss.write_index(index, INDEX_PATH)

    print("FAISS index saved to:")
    print(INDEX_PATH)


def load_resources():

    print("Loading FAISS index...")

    index = faiss.read_index(INDEX_PATH)

    print("Loading chunks...")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    return index, chunks, model


def search(query, index, chunks, model, top_k=5):

    start = time.perf_counter()

    # Convert query into embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Retrieve more candidates first
    candidate_k = min(20, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        candidate_k
    )

    results = []
    seen_texts = set()

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        text = chunks[idx]["text"].strip()

        # Remove exact duplicate chunks
        if text in seen_texts:
            continue

        seen_texts.add(text)

        result = {
            "score": float(score),
            "text": text,
            "strategy": chunks[idx].get(
                "strategy",
                "unknown"
            )
        }

        results.append(result)

        # Keep only final top-k
        if len(results) >= top_k:
            break

    latency = (
        time.perf_counter() - start
    ) * 1000

    return results, latency
def retrieve(query, top_k=5):

    index, chunks, model = load_resources()

    results, latency = search(
        query,
        index,
        chunks,
        model,
        top_k=top_k
    )

    return {
        "query": query,
        "results": results,
        "latency_ms": round(latency, 2)
    }

def main():

    # Build index if it doesn't exist
    if not os.path.exists(INDEX_PATH):

        print("FAISS index not found.")
        print("Building index...\n")

        build_index()

    index, chunks, model = load_resources()

    print("\nRAG Retrieval Test")
    print("------------------")

    while True:

        query = input(
            "\nEnter your question "
            "(or type 'exit'): "
        )

        if query.lower() == "exit":
            break

        results, latency = search(
            query,
            index,
            chunks,
            model,
            top_k=5
        )

        print(
            f"\nRetrieval latency: "
            f"{latency:.2f} ms"
        )

        print("\nTop results:\n")

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"Result {i} "
                f"(score={result['score']:.4f}, "
                f"strategy={result['strategy']})"
            )

            print(result["text"][:500])
            print("-" * 60)


if __name__ == "__main__":
    main()