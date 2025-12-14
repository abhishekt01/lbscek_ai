"""
Utility functions for the voice assistant
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

def setup_logging(log_file: str = "assistant.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_config() -> Dict[str, Any]:
    """Get application configuration"""
    return {
        "app_name": "സർവജ്ഞ (Sarva-jña)",
        "version": "1.0.0",
        "max_conversation_history": 50,
        "supported_languages": ["en", "ml", "manglish"],
        "tts_enabled": True,
        "debug_mode": os.getenv("DEBUG", "false").lower() == "true"
    }

def validate_environment():
    """Validate required environment variables"""
    required_vars = ["PPLX_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing_vars)}"
        )

def format_timestamp(timestamp: datetime = None) -> str:
    """Format timestamp for display"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%I:%M %p, %b %d")

def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Truncate if too long
    if len(text) > 1000:
        text = text[:1000] + "..."
    
    return text.strip()