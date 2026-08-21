import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/validation/hinval.parquet"

# Multilingual model — works with Hindi + English
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_dataset():
    return pd.read_parquet(DATA_PATH)


def fixed_chunk(text, chunk_size=500, overlap=100):
    """Fixed-size chunking with overlap."""
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def sentence_chunk(text):
    """Hindi sentence-aware chunking."""
    if not text:
        return []

    sentences = [
        s.strip()
        for s in text.replace("।", "।|").split("|")
        if s.strip()
    ]

    return sentences


def semantic_chunk(text, model, threshold=0.55):
    """
    Semantic chunking:
    Groups consecutive sentences when they are semantically similar.
    """

    sentences = sentence_chunk(text)

    if len(sentences) <= 1:
        return sentences

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True
    )

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):

        similarity = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        if similarity >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def passage_chunk(passages):
    """Keep each original passage as a chunk."""
    if passages is None:
        return []

    return [
        str(p).strip()
        for p in passages
        if str(p).strip()
    ]


def create_chunks(row, model):
    """
    Generate multiple chunking strategies
    while preserving metadata.
    """

    passages = row["passages"]["Translated_passages"]

    chunks = []

    for passage_id, passage in enumerate(passages):

        # ------------------------------------------------
        # 1. PASSAGE LEVEL
        # ------------------------------------------------

        chunks.append({
            "text": passage,
            "strategy": "passage",
            "query_id": row["query_id"],
            "passage_id": passage_id,
            "language": row["target_lang"]
        })

        # ------------------------------------------------
        # 2. SENTENCE LEVEL
        # ------------------------------------------------

        sentences = sentence_chunk(passage)

        for sentence_id, sentence in enumerate(sentences):

            chunks.append({
                "text": sentence,
                "strategy": "sentence",
                "query_id": row["query_id"],
                "passage_id": passage_id,
                "sentence_id": sentence_id,
                "language": row["target_lang"]
            })

        # ------------------------------------------------
        # 3. FIXED SIZE + OVERLAP
        # ------------------------------------------------

        fixed_chunks = fixed_chunk(passage)

        for chunk_id, chunk in enumerate(fixed_chunks):

            chunks.append({
                "text": chunk,
                "strategy": "fixed_overlap",
                "query_id": row["query_id"],
                "passage_id": passage_id,
                "chunk_id": chunk_id,
                "language": row["target_lang"]
            })

        # ------------------------------------------------
        # 4. SEMANTIC CHUNKING
        # ------------------------------------------------

        semantic_chunks = semantic_chunk(
            passage,
            model
        )

        for chunk_id, chunk in enumerate(semantic_chunks):

            chunks.append({
                "text": chunk,
                "strategy": "semantic",
                "query_id": row["query_id"],
                "passage_id": passage_id,
                "chunk_id": chunk_id,
                "language": row["target_lang"]
            })

    return chunks


if __name__ == "__main__":

    print("Loading dataset...")

    df = load_dataset()

    print(f"Loaded {len(df)} records.")

    print("\nLoading multilingual embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded.")

    # Test only first record
    chunks = create_chunks(df.iloc[0], model)

    print(f"\nGenerated {len(chunks)} chunks.")

    strategies = {}

    for chunk in chunks:
        strategies.setdefault(
            chunk["strategy"], 0
        )
        strategies[chunk["strategy"]] += 1

    print("\nChunking strategy counts:")

    for strategy, count in strategies.items():
        print(f"{strategy}: {count}")

    print("\nExample semantic chunks:")

    semantic_examples = [
        c for c in chunks
        if c["strategy"] == "semantic"
    ]

    for i, chunk in enumerate(semantic_examples[:5]):

        print(f"\nSemantic Chunk {i + 1}:")
        print(chunk["text"][:500])