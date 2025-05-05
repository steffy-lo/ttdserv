import asyncio
from googletrans import Translator

translator = Translator()

async def translate_text(text: str, target_language: str = 'en', source_language: str = 'en') -> str:
    translation = await translator.translate(text, dest=target_language, src=source_language)
    return translation.text