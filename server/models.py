from pydantic import BaseModel
from typing import List, Annotated, Optional, Any
from annotated_types import Len

class FileInput(BaseModel):
    object_key: str

class TranscribeFileInput(BaseModel):
    object_key: str
    target_language: str
    model: str
    task: str

class Transcription(BaseModel):
    start: float
    end: float
    text: str
    speaker: str

class TranscriptionRequest(BaseModel):
    transcriptions: List[Annotated[list[Any], Len(4, 4)]] # start, end, text, speaker
    target_language: str
    source_language: str

class ProcessAudioRequest(TranscribeFileInput):
    target_language: str
    model: Optional[str] = "whisper"
    task: Optional[str] = "transcribe"
    preprocess: Optional[bool] = False