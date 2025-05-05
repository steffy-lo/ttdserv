import whisperx
from pyannote.audio import Pipeline
from speechlib import Transcriptor
import os

# Load the pyannote.audio speaker diarization pipeline
HF_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face access token not found. Please set the HUGGINGFACE_ACCESS_TOKEN environment variable.")

diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)

NO_WHITESPACE_LANGS = ["ja", "ko", "zh"]

LOG_FOLDER = "logs"       # log folder for storing transcripts
VOICE_FOLDER = "voices"   # folder containing subfolders named after each speaker with speaker voice samples in them. This will be used for speaker recognition   
MODEL_SIZE = "large"     # size of model to be used [tiny, small, medium, large-v1, large-v2, large-v3]
ACCESS_TOKEN = HF_TOKEN   # huggingface access token

def transcriptor(file, target_language):
    return Transcriptor(file, LOG_FOLDER, target_language, MODEL_SIZE, ACCESS_TOKEN)


device = "cpu" 
batch_size = 8 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

def format_whisperX_result(result):
    formatted_result = []
    # Iterate through each segment
    for segment in result["segments"]:
        seg = [
            segment["start"],
            segment["end"],
            segment["text"],
            segment["speaker"],
        ]
        formatted_result.append(seg)
    return formatted_result

def transcribe_with_whisperX(audio_file, task=None):
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    print("WhisperX model loaded, parsing audio...", flush=True)
    audio = whisperx.load_audio(audio_file)
    print("Transcribing audio...", flush=True)
    result = model.transcribe(audio, batch_size=batch_size, task=task)
    target_language = result["language"]

    # 2. Align whisper output
    print("Loading whisperX align model...", flush=True)
    model_a, metadata = whisperx.load_align_model(language_code=target_language, device=device)
    print("Aligning whisperX output...", flush=True)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3. Assign speaker labels
    print("Assigning speaker labels...", flush=True)
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
    diarize_segments = diarize_model(audio)
    print("Assigning word speaker labels...", flush=True)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return format_whisperX_result(result)
