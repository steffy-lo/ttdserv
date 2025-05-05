import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
import asyncio
import uuid
from typing import Optional, Any

from models import FileInput, TranscribeFileInput, TranscriptionRequest, Transcription
from workflows.transcribe import transcribe, diarize, transcribe_with_diarization
from workflows.denoise import denoise
from workflows.task import process_audio_file
from workflows.translate import translate_text
from lib.aws import file_upload

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

@app.get("/")
def root():
    return { "status": "OK" }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return file_upload(file)

@app.post("/denoise")
async def denoise_file(input: FileInput):
    return denoise(input.object_key)

@app.post("/transcribe")
async def transcribe_audio(input: TranscribeFileInput):
    """Transcribes an audio file from S3, uploads the transcription to S3, and returns S3 key."""
    try:
        transcription = transcribe(input.object_key, input.target_language)
        return { "transcription": transcription }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    

@app.post("/diarize")
async def diarize_api(input: FileInput):
    """Diarizes an audio file from S3, saves the diarization data to S3, and returns the S3 key."""
    try:
        diarization = diarize(input.object_key)
        # Convert diarization object to a serializable format (e.g., list of dictionaries)
        diarization_results = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            diarization_results.append({
                "start_time": segment.start,
                "end_time": segment.end,
                "speaker": speaker
            })

        return {"diarization": diarization_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diarization failed: {e}")
    
@app.post("/transcribe_with_diarization")
async def transcribe_and_diarize_api(input: TranscribeFileInput):
    """Transcribes and diarizes an audio file.  Downloads existing transcription and diarization from S3 if they exist. Uploads combined results to S3."""
    try:
        result = transcribe_with_diarization(input.object_key, input.target_language, model=input.model, task=input.task)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcribe and Diarization failed: {e}")
    

@app.post("/translate")
async def translate_transcriptions(request: TranscriptionRequest) -> Any:
    try:
        translated_text = await asyncio.gather(*[translate_text(transcription[2], request.target_language, request.source_language) for transcription in request.transcriptions])
        translated_transcriptions = [
            Transcription(
                start=transcription[0],
                end=transcription[1],
                text=translated_text[i],
                speaker=transcription[3]
            )
            for i, transcription in enumerate(request.transcriptions)
        ]
        
        return { "transcriptions": translated_transcriptions }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_audio(file: UploadFile = File(...), target_language: Optional[str] = Form(None), model: Optional[str] = Form("whisper"), task: Optional[str] = Form("transcribe"), preprocess: Optional[bool] = Form(False)):
    """
    Endpoint to upload audio and initiate processing.
    """

    # Generate a unique ID for the request
    request_id = str(uuid.uuid4())
    response = process_audio_file(request_id, file, {
        "target_language": target_language, 
        "model": model, 
        "task": task,
        "preprocess": preprocess
    })
    return {"request_id": request_id, "data": response }
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)