from fastapi import HTTPException
from lib.aws import file_download, upload_to_s3
import noisereduce as nr
import soundfile as sf
import io

def denoise(object_key):
    # Download and load the audio data from S3
    try:
        audio_data = file_download(object_key)
        audio_bytes = io.BytesIO(audio_data)
        data, sample_rate = sf.read(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading and reading audio file: {e}")

    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=sample_rate)

    # Save the denoised audio to a BytesIO object
    denoised_audio = io.BytesIO()
    sf.write(denoised_audio, reduced_noise, sample_rate, format='WAV')
    denoised_audio.seek(0)

    # Define S3 object name
    denoised_object_name = f"denoised/{object_key}"

    # Upload denoised audio to S3
    upload_to_s3(denoised_audio, denoised_object_name)

    return { "message": "Denoised audio uploaded to S3", "object_key": denoised_object_name }
