import os
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise ValueError("SARVAM_API_KEY not found in .env")

client = SarvamAI(api_subscription_key=api_key)


def transcribe_audio(audio_path: str) -> str:
    """
    Convert an audio file into text using Sarvam AI.
    """

    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    with open(path, "rb") as audio_file:
        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            mode="transcribe"
        )

    return response.transcript


if __name__ == "__main__":
    audio_path = "audio test.mp3.wav"
    transcript = transcribe_audio(audio_path)

    print("Transcript:")
    print(transcript)