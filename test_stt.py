import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise ValueError("SARVAM_API_KEY not found in .env")

client = SarvamAI(api_subscription_key=api_key)

with open("audio test.mp3", "rb") as audio_file:
    response = client.speech_to_text.transcribe(
        file=audio_file,
        model="saaras:v3",
        mode="transcribe"
    )

print("Transcript:")
print(response.transcript)