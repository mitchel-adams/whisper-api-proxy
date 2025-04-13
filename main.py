from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
import base64
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class TranscribeRequest(BaseModel):
    audio_base64: str

@app.post("/transcribe")
async def transcribe_audio(req: TranscribeRequest):
    audio_data = base64.b64decode(req.audio_base64.split(",")[-1])  # remove metadata if present
    file = ("audio.m4a", BytesIO(audio_data), "audio/m4a")

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            files={"file": file},
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            data={"model": "whisper-1"},
        )

    result = r.json()
    return { "transcript": result["text"] }
