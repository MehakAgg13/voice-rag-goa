from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os

from backend.rag_pipeline import process_audio

app = FastAPI(title="Voice RAG API")


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Voice RAG API is running"
    }


@app.post("/process")
async def process(file: UploadFile = File(...)):

    # Basic file validation
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file provided"
        )

    # Save uploaded audio temporarily
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        contents = await file.read()
        temp_file.write(contents)
        temp_path = temp_file.name

    try:
        # Run complete voice RAG pipeline
        result = process_audio(temp_path)

        return {
            "transcript": result["transcript"],
            "answer": result["answer"]
        }

    finally:
        # Delete temporary audio file
        if os.path.exists(temp_path):
            os.remove(temp_path)