import os

from dotenv import load_dotenv
from google import genai
from backend.guardrails import check_query, check_grounding

# Load variables from .env
load_dotenv()

# Get Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Create Gemini client
client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.6-flash"


def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer using Gemini based only on retrieved context.
    """

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer,
say:

"I don't know based on the provided context."

Do not make up facts.

Context:
{context}

User question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()

    from backend.guardrails import check_query, check_grounding


def generate_safe_answer(query: str, retrieved_chunks: list) -> str:
    """
    Run guardrails before sending the query to Gemini.
    """

    # 1. Check query
    guardrail_result = check_query(query)

    if not guardrail_result["allowed"]:
        return guardrail_result["message"]

    # 2. Check retrieved context
    grounding_result = check_grounding(retrieved_chunks)

    if not grounding_result["grounded"]:
        return grounding_result["message"]

    # 3. Prepare context
    context_parts = []

    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
        else:
            text = str(chunk)

        if text:
            context_parts.append(text)

    context = "\n\n".join(context_parts)

    # 4. Generate answer using Gemini
    return generate_answer(query, context)