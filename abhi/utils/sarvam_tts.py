import os
import requests
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


class SarvamTTS:
    def __init__(self):
        if not SARVAM_API_KEY:
            raise ValueError("SARVAM_API_KEY not found in environment")
        self.url = "https://api.sarvam.ai/text-to-speech"
        self.api_key = SARVAM_API_KEY

    def synthesize(self, text: str, lang_code: str = "ml-IN", speaker: str = "manisha"):
        """
        Returns raw audio bytes (MP3) from Sarvam TTS.
        """
        payload = {
            "inputs": [text],
            "target_language_code": lang_code,
            "speaker": speaker,
            "model": "bulbul:v2",
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        resp = requests.post(self.url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        # API may return JSON with base64 or direct bytes; here assume bytes for simplicity.
        # If response is JSON with 'audio' field, adapt accordingly from official docs.
        return resp.content
