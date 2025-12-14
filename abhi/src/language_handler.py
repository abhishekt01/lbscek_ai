"""
Language detection and translation module
"""

from langdetect import detect, DetectorFactory
from ml2en import ml2en
import logging

# For consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

class LanguageHandler:
    """Handles language detection and conversion"""
    
    def __init__(self):
        self.malayalam_unicode_range = ("\u0d00", "\u0d7f")
    
    def detect_language_mode(self, text: str) -> str:
        """
        Detect language mode of input text
        
        Returns:
            - 'en': English
            - 'ml_script': Malayalam Unicode
            - 'manglish': Malayalam in English letters / mixed
        """
        text = text.strip()
        if not text:
            return "en"
        
        # Check for Malayalam Unicode characters
        if any(self.malayalam_unicode_range[0] <= ch <= self.malayalam_unicode_range[1] 
               for ch in text):
            return "ml_script"
        
        # Try langdetect for English
        try:
            lang_code = detect(text)
            if lang_code == "en":
                return "en"
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
        
        # Default to manglish
        return "manglish"
    
    def malayalam_to_manglish(self, malayalam_text: str) -> str:
        """
        Convert Malayalam script to Manglish (English letters)
        
        Args:
            malayalam_text: Text in Malayalam script
            
        Returns:
            Text in Manglish (English transliteration)
        """
        try:
            return ml2en(malayalam_text)
        except Exception as e:
            logger.error(f"Malayalam to Manglish conversion failed: {e}")
            return malayalam_text  # Return original if conversion fails
    
    def get_language_name(self, mode: str) -> str:
        """Get human-readable language name"""
        language_names = {
            "en": "English",
            "ml_script": "Malayalam",
            "manglish": "Manglish (Malayalam in English letters)"
        }
        return language_names.get(mode, "Unknown")