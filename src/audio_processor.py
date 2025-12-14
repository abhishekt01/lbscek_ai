"""
Audio processing module for voice assistant
"""

import io
from typing import Optional
from gtts import gTTS
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles text-to-speech and audio processing"""
    
    def __init__(self):
        self.supported_languages = {
            "en": "English",
            "ml": "Malayalam",
            "hi": "Hindi",
            "ta": "Tamil"
        }
    
    def text_to_speech(self, text: str, lang_code: str = "ml") -> Optional[bytes]:
        """
        Convert text to speech using gTTS
        
        Args:
            text: Text to convert
            lang_code: Language code (en, ml, etc.)
            
        Returns:
            Audio bytes in MP3 format or None on error
        """
        try:
            if lang_code not in self.supported_languages:
                lang_code = "en"  # Default to English
            
            tts = gTTS(text=text, lang=lang_code, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.read()
            
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            return None
    
    def validate_audio_length(self, text: str, max_length: int = 1000) -> bool:
        """
        Validate if text is suitable for TTS
        
        Args:
            text: Text to validate
            max_length: Maximum allowed characters
            
        Returns:
            True if valid, False otherwise
        """
        return len(text) <= max_length