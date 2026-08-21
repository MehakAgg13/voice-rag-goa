import os
import time

from google import genai
from backend.retrieval import retrieve


MODEL_NAME = "gemini-3.6-flash"

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)


def generate_answer(query, retrieved_results):

    context = "\n\n".join(
        [
            f"[Source {i + 1} | Strategy: {result['strategy']}]\n"
            f"{result['text']}"
            for i, result in enumerate(retrieved_results)
        ]
    )

    prompt = f"""
You are a grounded multilingual RAG assistant.

Answer the user's question ONLY using the retrieved context.

Rules:
1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent or assume facts.
4. If the context does not contain enough information, say:
   "मुझे उपलब्ध जानकारी में इसका उत्तर नहीं मिला।"
5. Answer in the same language as the user's question.
6. Keep the answer concise and clear.

User question:
{query}

Retrieved context:
{context}

Answer:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()


def rag_pipeline(query):

    start = time.perf_counter()

    retrieval_output = retrieve(query)

    retrieval_results = retrieval_output["results"]

    retrieval_latency = retrieval_output["latency_ms"]

    generation_start = time.perf_counter()

    answer = generate_answer(
        query,
        retrieval_results
    )

    generation_latency = (
        time.perf_counter() - generation_start
    ) * 1000

    total_latency = (
        time.perf_counter() - start
    ) * 1000

    return {
        "query": query,
        "answer": answer,
        "retrieval_latency_ms": retrieval_latency,
        "generation_latency_ms": generation_latency,
        "total_latency_ms": total_latency
    }


if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    )

    result = rag_pipeline(query)

    print("\n" + "=" * 60)
    print("RAG RESULT")
    print("=" * 60)

    print("\nQuestion:")
    print(result["query"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nLatency:")
    print(
        f"Retrieval: "
        f"{result['retrieval_latency_ms']:.2f} ms"
    )

    print(
        f"Generation: "
        f"{result['generation_latency_ms']:.2f} ms"
    )

    print(
        f"Total: "
        f"{result['total_latency_ms']:.2f} ms"
    )