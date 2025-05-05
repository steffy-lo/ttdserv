import asyncio
from models import ProcessAudioRequest
from lib.aws import file_upload
from workflows.denoise import denoise
from workflows.transcribe import transcribe_with_diarization

async def transcribe_audio(request_id: str):
    print(f"Transcribing audio: request_id={request_id}")
    await asyncio.sleep(5) 
    # Call transcribe api here
    return "Transcription result"

async def diarize_audio(request_id: str):
    print(f"Diarizing audio: request_id={request_id}")
    await asyncio.sleep(5)
    # Call diarize api here
    return "Diarization result"

def process_audio_file(request_id: str, file, request_data: ProcessAudioRequest):
    """
    Simulates the audio processing steps with progress updates.
    """
    try:
        print(f"Uploading audio file: request_id={request_id} request_data={request_data}", flush=True)
        progress = 10
        print(f"Progress: {progress} - Uploading file...", flush=True)
        upload_result = file_upload(file)
        object_key = upload_result["object_key"]

        if request_data["preprocess"]:
            print(f"Starting audio processing task: request_id={request_id}", flush=True)
            progress = 20
            print(f"Progress: {progress} - Denoising...", flush=True)
            denoise_result = denoise(object_key)
            object_key = denoise_result["object_key"]

        progress = 50
        print(f"Progress: {progress} - Transcribing and translating... Please be patient, this might take a while...", flush=True)
        result = transcribe_with_diarization(object_key, request_data["target_language"], model=request_data["model"], task=request_data["task"])

        progress = 100
        print(f"Progress: {progress} - Aggregating results... Completed!", request_id)
        return result

    except Exception as e:
        print(f"Error processing audio: request_id={request_id}, error: {e}")