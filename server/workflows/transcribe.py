from fastapi import HTTPException
from lib.aws import file_download, upload_to_s3
from lib.transcribe import diarization_pipeline, transcriptor, transcribe_with_whisperX
import whisper
import io
import os
import json

# Load the Whisper model
whisper_model = whisper.load_model("base")

TEMP_AUDIO_FILE = "temp_audio.mp3"
TEMP_AUDIO_FILE_WAV = "temp_audio.wav"

def _download_audio(object_key):
    """Downloads audio from S3 and saves it to a temporary file."""
    try:
        audio_data = file_download(object_key)
        audio_bytes = io.BytesIO(audio_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading audio file: {e}")

    audio_bytes.seek(0)
    with open(TEMP_AUDIO_FILE, "wb") as f:
        f.write(audio_bytes.read())

def transcribe(object_key, target_language):
    """Transcribes audio using Whisper."""
    try:
        _download_audio(object_key)  # Download audio to temp file
        transcription = whisper_model.transcribe(TEMP_AUDIO_FILE, task="translate", language=target_language)
        # Save the transcription to S3
        transcription_object_name = f"transcriptions/{os.path.splitext(object_key)[0]}_transcription.txt"
        transcription_bytes = io.BytesIO(transcription.encode('utf-8'))
        transcription_bytes.seek(0)
        upload_to_s3(transcription_bytes, transcription_object_name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transcription: {e}")
    finally:
        # Clean up temporary file, even if an error occurs.
        if os.path.exists(TEMP_AUDIO_FILE):
            os.remove(TEMP_AUDIO_FILE)

    return {"message": "Transcription uploaded to S3", "s3_key": transcription_object_name, "transcription": transcription["text"] }

def diarize(object_key):
    """Performs speaker diarization on the audio."""
    try:
        _download_audio(object_key) # Download audio to temp file
        diarization = diarization_pipeline(TEMP_AUDIO_FILE)
        # Convert diarization object to a serializable format (e.g., list of dictionaries)
        diarization_results = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            diarization_results.append({
                "start_time": segment.start,
                "end_time": segment.end,
                "speaker": speaker
            })

        # Save the diarization results to S3 (JSON format)
        diarization_object_name = f"diarizations/{os.path.splitext(object_key)[0]}_diarization.json"
        diarization_bytes = io.BytesIO(json.dumps(diarization_results).encode('utf-8')) # Serialize to JSON
        diarization_bytes.seek(0)
        upload_to_s3(diarization_bytes, diarization_object_name)  # Assuming upload_to_s3 exists in your server.lib.aws

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during speaker diarization: {e}")
    finally:
        # Clean up temporary file, even if an error occurs.
        if os.path.exists(TEMP_AUDIO_FILE):
            os.remove(TEMP_AUDIO_FILE)

    return { "message": "Diarization data uploaded to S3", "s3_key": diarization_object_name, "diarization": diarization_results }
    
def transcribe_with_diarization(object_key, target_language, model="whisper", task=None):
    _download_audio(object_key)  # Download audio to temp file

    if model == "whisper":
        print("Using whisper to transcribe...", flush=True)
        results = transcriptor(TEMP_AUDIO_FILE, target_language).whisper()
    elif model == "whisperX":
        print("Using whisperX to transcribe...", flush=True)
        results = transcribe_with_whisperX(TEMP_AUDIO_FILE, task=task)

    # Save the results to S3 (JSON format)
    object_name = f"transcriptions/{os.path.splitext(object_key)[0]}_transcription.json"
    bytes = io.BytesIO(json.dumps(results).encode('utf-8')) # Serialize to JSON
    bytes.seek(0)
    upload_to_s3(bytes, object_name)
    # Delete the temporary file after closing, remove both .mp3 and .wav
    try:
        os.remove(TEMP_AUDIO_FILE)
        os.remove(TEMP_AUDIO_FILE_WAV)
    except Exception as e:
        print(f"Error deleting temp file: {e}")
    return {"message": "Transcription with speaker diarization uploaded to S3", "s3_key": object_name, "result": results }

