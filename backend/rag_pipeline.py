from backend.retrieval import retrieve
from backend.llm import generate_safe_answer
from backend.guardrails import check_query
from backend.speech import transcribe_audio


def process_query(transcript: str) -> dict:
    """
    Complete RAG pipeline:

    Transcript
        ↓
    Input guardrail
        ↓
    Retrieval
        ↓
    Grounding guardrail
        ↓
    Gemini
        ↓
    Answer
    """

    # 1. Input guardrail
    query_check = check_query(transcript)

    if not query_check["allowed"]:
        return {
            "transcript": transcript,
            "answer": query_check["message"],
            "retrieved_chunks": [],
            "retrieval_latency_ms": 0
        }

    # 2. Retrieve relevant chunks
    retrieval_result = retrieve(
        transcript,
        top_k=5
    )

    retrieved_chunks = retrieval_result["results"]

    # 3. Grounding guardrail + Gemini
    answer = generate_safe_answer(
        transcript,
        retrieved_chunks
    )

    return {
        "transcript": transcript,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_latency_ms": retrieval_result["latency_ms"]
    }
def process_audio(audio_path: str) -> dict:

    transcript = transcribe_audio(audio_path)

    result = process_query(transcript)

    return result



# if __name__ == "__main__":

#     print("RAG Pipeline Test")
#     print("-----------------")

#     query = input("Enter your question: ")

#     result = process_query(query)

#     print("\nTranscript:")
#     print(result["transcript"])

#     print("\nAnswer:")
#     print(result["answer"])

#     print(
#         f"\nRetrieval latency: "
#         f"{result['retrieval_latency_ms']} ms"
#     )

if __name__ == "__main__":

    audio_path = "audio test.mp3.wav"

    result = process_audio(audio_path)

    print("\nTranscript:")
    print(result["transcript"])

    print("\nAnswer:")
    print(result["answer"])

    print(
        f"\nRetrieval latency: "
        f"{result['retrieval_latency_ms']} ms"
    )