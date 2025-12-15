
import os
import json
import difflib
import io
import base64
from datetime import datetime
from typing import Optional, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from ml2en import ml2en
from openai import OpenAI
from streamlit_mic_recorder import speech_to_text
from sarvamai import SarvamAI

# For consistent language detection
DetectorFactory.seed = 0

# -------------------------------
# LOAD ENVIRONMENT
# -------------------------------
load_dotenv()

PPLX_API_KEY = os.getenv("PPLX_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not PPLX_API_KEY:
    st.error("⚠️ PPLX_API_KEY is missing! Please create a .env file with your API key.")
    st.stop()

if not SARVAM_API_KEY:
    st.error("⚠️ SARVAM_API_KEY is missing! Please add it to your .env file.")
    st.stop()

# -------------------------------
# INITIALIZE CLIENTS
# -------------------------------
pplx_client = OpenAI(
    api_key=PPLX_API_KEY,
    base_url="https://api.perplexity.ai",
)

sarvam_client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY,
)

# -------------------------------
# KNOWLEDGE BASE CLASS
# -------------------------------
class KnowledgeBase:
    def __init__(self, file_name: str = "faq_data.json"):
        self.file_path = file_name
        self.faqs = []
        self.load_faqs()
    
    def load_faqs(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.faqs = json.load(f)
            elif os.path.exists(f"data/{self.file_path}"):
                with open(f"data/{self.file_path}", "r", encoding="utf-8") as f:
                    self.faqs = json.load(f)
            else:
                st.warning(f"Knowledge base file not found.")
                self.faqs = []
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            self.faqs = []
    
    def _normalize(self, text: str) -> str:
        return text.lower().strip()
    
    def get_relevant_info(self, query: str) -> Optional[Dict[str, Any]]:
        if not self.faqs:
            return None
        
        q_norm = self._normalize(query)
        
        for entry in self.faqs:
            for pattern in entry.get("question_patterns", []):
                p_norm = self._normalize(pattern)
                if p_norm in q_norm or q_norm in p_norm:
                    return entry
        
        all_tags = []
        tag_to_entry = {}
        for entry in self.faqs:
            for tag in entry.get("tags", []):
                t_norm = self._normalize(tag)
                all_tags.append(t_norm)
                tag_to_entry[t_norm] = entry
        
        close_matches = difflib.get_close_matches(q_norm, all_tags, n=1, cutoff=0.5)
        if close_matches:
            return tag_to_entry[close_matches[0]]
        
        return None

# -------------------------------
# LANGUAGE HANDLER
# -------------------------------
class LanguageHandler:
    def __init__(self):
        self.malayalam_unicode_range = ("\u0d00", "\u0d7f")
    
    def detect_language_mode(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "en"
        
        if any(self.malayalam_unicode_range[0] <= ch <= self.malayalam_unicode_range[1] for ch in text):
            return "ml_script"
        
        try:
            lang_code = detect(text)
            if lang_code == "en":
                return "en"
        except Exception:
            pass
        
        return "manglish"
    
    def malayalam_to_manglish(self, malayalam_text: str) -> str:
        try:
            return ml2en(malayalam_text)
        except Exception:
            return malayalam_text

# -------------------------------
# AI PROCESSOR
# -------------------------------
class AIProcessor:
    def __init__(self, model: str = "sonar"):
        self.model = model
        self.client = pplx_client
    
    def rewrite_from_kb(self, user_query: str, kb_entry: Dict[str, Any], lang_mode: str) -> str:
        facts = kb_entry.get("answer_facts", {})
        tags = kb_entry.get("tags", [])
        
        kb_text = "\n".join(f"{key}: {value}" for key, value in facts.items())
        
        if lang_mode == "en":
            lang_instruction = "Answer in simple, clear English. Use a friendly, helpful tone suitable for college students."
        else:
            lang_instruction = "Answer in Manglish (Malayalam written in English letters). Use a warm, natural, conversational tone."
        
        system_prompt = f"""You are സർവജ്ഞ (Sarva-jña), an AI assistant for LBS College of Engineering, Kasaragod.

CRITICAL RULES:
1. Use ONLY the information provided in KB_FACTS
2. Do NOT invent new facts or details
3. Do NOT use internet knowledge
4. Structure your response naturally, not as a list
5. {lang_instruction}

If KB_FACTS don't fully answer the question, acknowledge this politely.
Keep responses under 300 words for better voice playback."""
        
        user_message = f"""User Question: {user_query}

Relevant Tags: {', '.join(tags)}

KB Facts:
{kb_text}

Please provide a helpful answer based only on the KB facts above."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"AI Processing Error: {e}")
            
            if lang_mode == "en":
                return f"Here's what I know: {' '.join(list(facts.values()))[:200]}"
            else:
                return f"ഇതാ എനിക്കറിയാവുന്നത്: {' '.join(list(facts.values()))[:200]}"

# -------------------------------
# SARVAM AI AUDIO PROCESSOR WITH AUTO-PLAY
# -------------------------------
class AudioProcessor:
    def __init__(self):
        self.client = sarvam_client
        self.supported_languages = {
            "en": "en-IN",
            "ml": "ml-IN",
            "hi": "hi-IN",
            "ta": "ta-IN",
            "te": "te-IN",
            "kn": "kn-IN",
            "bn": "bn-IN",
            "gu": "gu-IN",
            "mr": "mr-IN",
            "od": "od-IN",
            "pa": "pa-IN"
        }
        self.speakers = {
            "ml": ["arya", "meera", "pavithra", "maitreyi"],
            "en": ["arya", "meera", "pavithra", "maitreyi"],
            "hi": ["arya", "meera", "pavithra", "maitreyi"],
            "default": ["arya", "meera", "pavithra", "maitreyi"]
        }
        self.audio_cache = {}
    
    def get_pace_value(self, speech_rate: str) -> float:
        """Convert speech rate setting to pace value for Sarvam AI"""
        pace_map = {
            "Slow": 0.8,
            "Normal": 1.0,
            "Fast": 1.2
        }
        return pace_map.get(speech_rate, 1.0)
    
    def text_to_speech(
        self, 
        text: str, 
        lang_code: str = "ml",
        speaker: str = "arya",
        pitch: float = 0,
        pace: float = 1.0,
        loudness: float = 1.0,
        sample_rate: int = 22050
    ) -> Optional[bytes]:
        """
        Convert text to speech using Sarvam AI TTS API
        
        Args:
            text: Text to convert to speech
            lang_code: Language code (ml, en, hi, etc.)
            speaker: Voice speaker name
            pitch: Pitch adjustment (-10 to 10)
            pace: Speech pace (0.5 to 2.0)
            loudness: Volume level (0.5 to 2.0)
            sample_rate: Audio sample rate (8000, 16000, 22050, 24000)
        
        Returns:
            Audio bytes or None if error
        """
        try:
            if not text or not text.strip():
                return None
            
            # Map language code to Sarvam AI format
            target_language = self.supported_languages.get(lang_code, "en-IN")
            
            # Create cache key
            cache_key = f"{hash(text)}_{target_language}_{speaker}_{pace}"
            
            if cache_key in self.audio_cache:
                return self.audio_cache[cache_key]
            
            # Call Sarvam AI TTS API
            response = self.client.text_to_speech.convert(
                text=text,
                target_language_code=target_language,
                speaker=speaker,
                pitch=pitch,
                pace=pace,
                loudness=loudness,
                speech_sample_rate=sample_rate,
                enable_preprocessing=True,
                model="bulbul:v2"
            )
            
            # Handle response - Sarvam AI returns base64 encoded audio
            audio_bytes = None
            
            if hasattr(response, 'audio'):
                # If response has audio attribute (base64 string)
                if isinstance(response.audio, str):
                    audio_bytes = base64.b64decode(response.audio)
                elif isinstance(response.audio, bytes):
                    audio_bytes = response.audio
            elif hasattr(response, 'audios') and response.audios:
                # If response has audios list
                audio_data = response.audios[0]
                if isinstance(audio_data, str):
                    audio_bytes = base64.b64decode(audio_data)
                elif isinstance(audio_data, bytes):
                    audio_bytes = audio_data
            elif isinstance(response, dict):
                # If response is a dictionary
                if 'audio' in response:
                    audio_bytes = base64.b64decode(response['audio'])
                elif 'audios' in response and response['audios']:
                    audio_bytes = base64.b64decode(response['audios'][0])
            elif isinstance(response, str):
                # If response is directly a base64 string
                audio_bytes = base64.b64decode(response)
            elif isinstance(response, bytes):
                # If response is directly bytes
                audio_bytes = response
            
            if audio_bytes:
                self.audio_cache[cache_key] = audio_bytes
                return audio_bytes
            else:
                st.warning("Could not extract audio from Sarvam AI response")
                return None
            
        except Exception as e:
            st.error(f"Sarvam AI TTS Error: {e}")
            return None
    
    def create_audio_autoplay_html(self, audio_bytes: bytes, autoplay: bool = True) -> str:
        """Create HTML audio element with autoplay support"""
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        autoplay_attr = "autoplay" if autoplay else ""
        html = f"""
        <audio id="autoPlayAudio" controls {autoplay_attr} style="width: 100%; margin-top: 10px;">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const audio = document.getElementById('autoPlayAudio');
            if (audio) {{
                const playPromise = audio.play();
                if (playPromise !== undefined) {{
                    playPromise.then(_ => {{
                        console.log('Audio autoplay started successfully');
                    }}).catch(error => {{
                        console.log('Autoplay prevented:', error);
                        audio.controls = true;
                    }});
                }}
            }}
        }});
        </script>
        """
        return html
    
    def get_available_speakers(self, lang_code: str = "ml") -> list:
        """Get available speakers for a language"""
        return self.speakers.get(lang_code, self.speakers["default"])

# -------------------------------
# SESSION STATE MANAGEMENT
# -------------------------------
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {
            "language": "auto",
            "voice_enabled": True,
            "tts_language": "ml",
            "auto_play": True,
            "speech_rate": "Normal",
            "speaker": "arya",
            "pitch": 0,
            "loudness": 1.0
        }
    
    if "knowledge_base" not in st.session_state:
        st.session_state.kb = KnowledgeBase("faq_data.json")
    
    if "language_handler" not in st.session_state:
        st.session_state.lh = LanguageHandler()
    
    if "ai_processor" not in st.session_state:
        st.session_state.ai = AIProcessor()
    
    if "audio_processor" not in st.session_state:
        st.session_state.ap = AudioProcessor()
    
    if "last_audio_data" not in st.session_state:
        st.session_state.last_audio_data = None
    
    if "should_autoplay" not in st.session_state:
        st.session_state.should_autoplay = False

# -------------------------------
# STREAMLIT APP CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="സർവജ്ഞ – LBS College AI Voice Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #2E86AB 0%, #1B4F72 100%);
    color: white;
    text-align: center;
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.user-message {
    background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #4CAF50;
    margin: 15px 0;
}
.assistant-message {
    background: linear-gradient(135deg, #F0F7FF 0%, #D4E6FF 100%);
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #2E86AB;
    margin: 15px 0;
}
.voice-input-section {
    background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
    padding: 25px;
    border-radius: 12px;
    border: 3px dashed #2E86AB;
    margin: 20px 0;
    text-align: center;
}
.autoplay-indicator {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    background: #FFD700;
    border-radius: 15px;
    font-size: 0.8rem;
    color: #333;
    margin-left: 10px;
}
.sarvam-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 12px;
    background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
    border-radius: 15px;
    font-size: 0.75rem;
    color: white;
    margin-left: 10px;
}
</style>
""", unsafe_allow_html=True)

initialize_session_state()

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def add_message(role: str, content: str, is_voice: bool = False, audio_bytes: bytes = None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    message = {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "is_voice": is_voice,
        "audio_bytes": audio_bytes,
        "is_latest": True
    }
    
    for msg in st.session_state.messages:
        msg["is_latest"] = False
    
    st.session_state.messages.append(message)
    
    if role == "assistant" and audio_bytes and st.session_state.user_preferences["auto_play"]:
        st.session_state.should_autoplay = True
        st.session_state.last_audio_data = audio_bytes

def process_query(query: str, is_voice_input: bool = False):
    with st.spinner("Processing your question..."):
        lang_mode = st.session_state.lh.detect_language_mode(query)
        
        if lang_mode == "ml_script":
            processed_query = st.session_state.lh.malayalam_to_manglish(query)
            if is_voice_input:
                st.success(f"🎤 Voice recognized and converted: {processed_query}")
        else:
            processed_query = query
        
        kb_entry = st.session_state.kb.get_relevant_info(processed_query)
        
        if kb_entry:
            response = st.session_state.ai.rewrite_from_kb(processed_query, kb_entry, lang_mode)
        else:
            if lang_mode == "en":
                response = "Sorry, I couldn't find that information in the college database."
            else:
                response = "ക്ഷമിക്കണം, ഈ ചോദ്യത്തിനുള്ള വിവരങ്ങൾ കോളേജ് ഡാറ്റാബേസിൽ ഇല്ല."
        
        audio_bytes = None
        if st.session_state.user_preferences["voice_enabled"]:
            tts_lang = st.session_state.user_preferences["tts_language"]
            speaker = st.session_state.user_preferences.get("speaker", "arya")
            pitch = st.session_state.user_preferences.get("pitch", 0)
            loudness = st.session_state.user_preferences.get("loudness", 1.0)
            pace = st.session_state.ap.get_pace_value(
                st.session_state.user_preferences.get("speech_rate", "Normal")
            )
            
            audio_bytes = st.session_state.ap.text_to_speech(
                text=response, 
                lang_code=tts_lang,
                speaker=speaker,
                pitch=pitch,
                pace=pace,
                loudness=loudness
            )
        
        add_message("user", query, is_voice=is_voice_input)
        add_message("assistant", response, audio_bytes=audio_bytes)

def clear_conversation():
    st.session_state.messages = []
    st.session_state.last_audio_data = None
    st.session_state.should_autoplay = False
    st.success("Conversation cleared!")

def create_autoplay_audio():
    if (st.session_state.should_autoplay and 
        st.session_state.last_audio_data and 
        st.session_state.user_preferences["auto_play"]):
        
        st.session_state.should_autoplay = False
        
        html = st.session_state.ap.create_audio_autoplay_html(
            st.session_state.last_audio_data,
            autoplay=True
        )
        st.markdown(html, unsafe_allow_html=True)
        st.markdown('<span class="autoplay-indicator">🔊 Auto-playing</span>', unsafe_allow_html=True)

# -------------------------------
# SIDEBAR COMPONENT
# -------------------------------
def create_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #2E86AB;">🎓 സർവജ്ഞ</h2>
            <p style="color: #666; font-size: 0.9rem;">LBS College AI Assistant</p>
            <span class="sarvam-badge">Powered by Sarvam AI</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("### ⚙️ Voice Settings (Sarvam AI)")
        
        auto_play = st.checkbox(
            "Auto-play voice responses 🔊",
            value=st.session_state.user_preferences.get("auto_play", True),
            help="Automatically play voice responses"
        )
        st.session_state.user_preferences["auto_play"] = auto_play
        
        voice_enabled = st.checkbox(
            "Enable voice responses",
            value=st.session_state.user_preferences.get("voice_enabled", True),
            help="Enable Sarvam AI text-to-speech"
        )
        st.session_state.user_preferences["voice_enabled"] = voice_enabled
        
        if voice_enabled:
            tts_lang = st.radio(
                "Response language",
                ["Malayalam", "English", "Hindi", "Tamil", "Telugu", "Kannada"],
                index=0 if st.session_state.user_preferences.get("tts_language") == "ml" else 1
            )
            lang_map = {
                "Malayalam": "ml",
                "English": "en",
                "Hindi": "hi",
                "Tamil": "ta",
                "Telugu": "te",
                "Kannada": "kn"
            }
            st.session_state.user_preferences["tts_language"] = lang_map.get(tts_lang, "ml")
            
            st.markdown("#### 🎙️ Voice Selection")
            speaker = st.selectbox(
                "Speaker voice",
                ["arya", "meera", "pavithra", "maitreyi"],
                index=["arya", "meera", "pavithra", "maitreyi"].index(
                    st.session_state.user_preferences.get("speaker", "arya")
                ),
                help="Select the voice for speech synthesis"
            )
            st.session_state.user_preferences["speaker"] = speaker
            
            speech_rate_options = ["Slow", "Normal", "Fast"]
            current_speech_rate = st.session_state.user_preferences.get("speech_rate", "Normal")
            
            if current_speech_rate not in speech_rate_options:
                current_speech_rate = "Normal"
            
            speech_rate = st.select_slider(
                "Speech speed",
                options=speech_rate_options,
                value=current_speech_rate
            )
            st.session_state.user_preferences["speech_rate"] = speech_rate
            
            st.markdown("#### 🎚️ Advanced Voice Settings")
            
            pitch = st.slider(
                "Pitch adjustment",
                min_value=-10.0,
                max_value=10.0,
                value=float(st.session_state.user_preferences.get("pitch", 0)),
                step=0.5,
                help="Adjust the pitch of the voice"
            )
            st.session_state.user_preferences["pitch"] = pitch
            
            loudness = st.slider(
                "Loudness",
                min_value=0.5,
                max_value=2.0,
                value=float(st.session_state.user_preferences.get("loudness", 1.0)),
                step=0.1,
                help="Adjust the volume level"
            )
            st.session_state.user_preferences["loudness"] = loudness
        
        st.divider()
        
        st.markdown("### 🌐 Language")
        language_option = st.selectbox(
            "Input language preference",
            ["Auto-detect", "English", "Malayalam", "Manglish"],
            index=0
        )
        
        language_map = {
            "Auto-detect": "auto",
            "English": "en",
            "Malayalam": "ml",
            "Manglish": "manglish"
        }
        st.session_state.user_preferences["language"] = language_map[language_option]
        
        st.divider()
        
        st.markdown("### 💬 Conversation")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_conversation()
                st.rerun()
        
        with col2:
            if st.button("🔊 Test Voice", use_container_width=True):
                test_text = "Hello! This is a test of the Sarvam AI voice assistant."
                if st.session_state.user_preferences["tts_language"] == "ml":
                    test_text = "നമസ്കാരം! ഇത് സർവം എഐ വോയ്സ് അസിസ്റ്റന്റിന്റെ ഒരു പരീക്ഷണമാണ്."
                
                audio_bytes = st.session_state.ap.text_to_speech(
                    text=test_text, 
                    lang_code=st.session_state.user_preferences["tts_language"],
                    speaker=st.session_state.user_preferences.get("speaker", "arya"),
                    pitch=st.session_state.user_preferences.get("pitch", 0),
                    pace=st.session_state.ap.get_pace_value(
                        st.session_state.user_preferences.get("speech_rate", "Normal")
                    ),
                    loudness=st.session_state.user_preferences.get("loudness", 1.0)
                )
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                else:
                    st.warning("Could not generate audio. Please check your Sarvam AI API key.")

# -------------------------------
# MAIN APP LAYOUT
# -------------------------------
def main():
    create_sidebar()
    
    st.markdown("""
    <div class="main-header">
        <h1>🎓 സർവജ്ഞ – LBS College AI Voice Assistant</h1>
        <p>Your Voice, Our Knowledge, Instant Answers in Malayalam & English</p>
        <p style="font-size: 1rem; background: rgba(255,255,255,0.2); padding: 8px; border-radius: 20px; display: inline-block;">
            🔊 Powered by <strong>Sarvam AI</strong> – Natural Indian Language TTS
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        auto_play_status = "🟢 ON" if st.session_state.user_preferences["auto_play"] else "🔴 OFF"
        st.markdown(f"**Auto-play:** {auto_play_status}")
    
    with col2:
        voice_status = "🟢 ON" if st.session_state.user_preferences["voice_enabled"] else "🔴 OFF"
        st.markdown(f"**Voice:** {voice_status}")
    
    with col3:
        lang_name = {
            "ml": "Malayalam",
            "en": "English",
            "hi": "Hindi",
            "ta": "Tamil",
            "te": "Telugu",
            "kn": "Kannada"
        }.get(st.session_state.user_preferences["tts_language"], "Malayalam")
        st.markdown(f"**Voice Lang:** {lang_name}")
    
    with col4:
        speaker = st.session_state.user_preferences.get("speaker", "arya").title()
        st.markdown(f"**Speaker:** {speaker}")
    
    with col5:
        kb_status = "🟢 Loaded" if st.session_state.kb.faqs else "🔴 Empty"
        st.markdown(f"**Knowledge:** {kb_status}")
    
    st.divider()
    
    col_main, col_sidebar = st.columns([3, 1])
    
    with col_main:
        tab1, tab2 = st.tabs(["💬 Text Input", "🎤 Voice Input"])
        
        with tab1:
            user_input = st.text_area(
                "Type your question:",
                placeholder="Example: What engineering courses are available?",
                height=100,
                key="text_input"
            )
            
            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                if st.button("📤 Submit & Speak", type="primary", use_container_width=True):
                    if user_input.strip():
                        process_query(user_input.strip())
                        st.rerun()
                    else:
                        st.warning("Please enter a question first!")
            
            with col_btn2:
                if st.button("🗑️ Clear Input", use_container_width=True):
                    st.session_state.text_input = ""
                    st.rerun()
        
        with tab2:
            st.markdown("""
            <div class="voice-input-section">
                <h3>🎤 Speak Your Question</h3>
                <p>Click below to record in Malayalam. The response will play automatically with Sarvam AI voice.</p>
            </div>
            """, unsafe_allow_html=True)
            
            voice_text = speech_to_text(
                language='ml-IN',
                start_prompt="🎙️ Start Recording",
                stop_prompt="⏹️ Stop",
                just_once=True,
                use_container_width=True,
                key="voice_recorder"
            )
            
            if voice_text:
                st.success("🎤 Voice input received!")
                st.info(f"**Recognized Text:** {voice_text}")
                process_query(voice_text, is_voice_input=True)
                st.rerun()
        
        st.divider()
        
        st.markdown("### 💭 Conversation")
        
        if not st.session_state.messages:
            st.info("👋 **Welcome to സർവജ്ഞ with Sarvam AI Voice!**\n\nAsk questions about LBS College in English, Malayalam, or Manglish.")
        
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                voice_icon = "🎤" if message.get('is_voice') else "💬"
                st.markdown(f"""
                <div class="user-message">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>👤 You {voice_icon}</strong>
                        <small style="color: #666;">{message['timestamp']}</small>
                    </div>
                    <p style="margin-top: 10px;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                autoplay_indicator = ""
                if message.get('is_latest') and st.session_state.user_preferences['auto_play']:
                    autoplay_indicator = '<span class="autoplay-indicator">🔊 Auto-play</span>'
                
                sarvam_badge = '<span class="sarvam-badge">Sarvam AI</span>'
                
                st.markdown(f"""
                <div class="assistant-message">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>🤖 സർവജ്ഞ</strong>
                            {sarvam_badge}
                            {autoplay_indicator}
                        </div>
                        <small style="color: #666;">{message['timestamp']}</small>
                    </div>
                    <p style="margin-top: 10px;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if (message.get('audio_bytes') and 
                    st.session_state.user_preferences["voice_enabled"]):
                    
                    st.audio(message['audio_bytes'], format="audio/wav")
                    
                    col_play, col_info = st.columns([1, 3])
                    with col_play:
                        if st.button(f"🔊 Play Again", key=f"play_{i}"):
                            st.session_state.should_autoplay = True
                            st.session_state.last_audio_data = message['audio_bytes']
                            st.rerun()
                    
                    with col_info:
                        lang_name = {
                            "ml": "Malayalam",
                            "en": "English",
                            "hi": "Hindi",
                            "ta": "Tamil",
                            "te": "Telugu",
                            "kn": "Kannada"
                        }.get(st.session_state.user_preferences["tts_language"], "Malayalam")
                        speaker = st.session_state.user_preferences.get("speaker", "arya").title()
                        speed = st.session_state.user_preferences['speech_rate']
                        st.caption(f"Voice: {lang_name} | Speaker: {speaker} | Speed: {speed}")
        
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            create_autoplay_audio()
    
    with col_sidebar:
        st.markdown("### ⚡ Quick Questions")
        
        quick_questions = [
            ("What engineering courses?", "courses"),
            ("Contact information?", "contact"),
            ("Library hours?", "library"),
            ("Admission process?", "admission"),
            ("Hostel facilities?", "hostel"),
            ("Placement details?", "placement"),
            ("College location?", "location"),
            ("എന്ത് കോഴ്സുകൾ?", "courses_ml"),
            ("ഫോൺ നമ്പർ?", "phone_ml"),
            ("ലൈബ്രറി സമയം?", "library_ml")
        ]
        
        for question_text, key in quick_questions:
            if st.button(question_text, key=f"quick_{key}", use_container_width=True):
                process_query(question_text)
                st.rerun()
        
        st.divider()
        
        st.markdown("### 🔊 Sarvam AI Voice Test")
        test_text = st.text_input("Test text:", "ഇത് ഒരു പരീക്ഷണമാണ്")
        
        col_test1, col_test2 = st.columns(2)
        with col_test1:
            if st.button("🇬🇧 English", use_container_width=True):
                audio_bytes = st.session_state.ap.text_to_speech(
                    text="This is a voice test using Sarvam AI.", 
                    lang_code="en",
                    speaker=st.session_state.user_preferences.get("speaker", "arya")
                )
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
        
        with col_test2:
            if st.button("🇮🇳 Malayalam", use_container_width=True):
                audio_bytes = st.session_state.ap.text_to_speech(
                    text=test_text if test_text else "നമസ്കാരം, ഇത് സർവം എഐ വോയ്സ് ടെസ്റ്റ് ആണ്.", 
                    lang_code="ml",
                    speaker=st.session_state.user_preferences.get("speaker", "arya")
                )
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
        
        st.divider()
        
        st.markdown("### 🎭 Voice Samples")
        st.caption("Try different Sarvam AI voices")
        
        for speaker_name in ["arya", "meera", "pavithra", "maitreyi"]:
            if st.button(f"🎙️ {speaker_name.title()}", key=f"sample_{speaker_name}", use_container_width=True):
                sample_text = "നമസ്കാരം, ഞാൻ സർവജ്ഞ ആണ്."
                audio_bytes = st.session_state.ap.text_to_speech(
                    text=sample_text,
                    lang_code="ml",
                    speaker=speaker_name
                )
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
    
    st.markdown("""
    <script>
    function handleAudioAutoplay() {
        const audioElements = document.querySelectorAll('audio');
        audioElements.forEach(audio => {
            if (audio.id === 'autoPlayAudio') {
                const playPromise = audio.play();
                if (playPromise !== undefined) {
                    playPromise.catch(error => {
                        console.log('Autoplay was prevented:', error);
                        audio.controls = true;
                    });
                }
            }
        });
    }
    document.addEventListener('DOMContentLoaded', handleAudioAutoplay);
    const observer = new MutationObserver(handleAudioAutoplay);
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; padding: 20px;">
        <p>🎓 <strong>സർവജ്ഞ – LBS College of Engineering, Kasaragod</strong></p>
        <p>🔊 <strong>Voice Assistant Powered by Sarvam AI (Bulbul v2)</strong></p>
        <p style="font-size: 0.8em; color: #999;">Natural Indian Language Text-to-Speech</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
