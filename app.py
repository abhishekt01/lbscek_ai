"""
സർവജ്ഞ (Sarva-jña) - LBS College AI Voice Assistant
A comprehensive voice-enabled assistant with auto-play voice responses
"""

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
from gtts import gTTS
from openai import OpenAI
from streamlit_mic_recorder import speech_to_text

# For consistent language detection
DetectorFactory.seed = 0

# -------------------------------
# LOAD ENVIRONMENT
# -------------------------------
load_dotenv()

PPLX_API_KEY = os.getenv("PPLX_API_KEY")
if not PPLX_API_KEY:
    st.error("⚠️ PPLX_API_KEY is missing! Please create a .env file with your API key.")
    st.stop()

# -------------------------------
# INITIALIZE CLIENTS
# -------------------------------
pplx_client = OpenAI(
    api_key=PPLX_API_KEY,
    base_url="https://api.perplexity.ai",
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
# AUDIO PROCESSOR WITH AUTO-PLAY
# -------------------------------
class AudioProcessor:
    def __init__(self):
        self.supported_languages = {
            "en": "English",
            "ml": "Malayalam"
        }
        self.audio_cache = {}
    
    def text_to_speech(self, text: str, lang_code: str = "ml") -> Optional[bytes]:
        try:
            if lang_code not in self.supported_languages:
                lang_code = "en"
            
            cache_key = f"{hash(text)}_{lang_code}"
            
            if cache_key in self.audio_cache:
                return self.audio_cache[cache_key]
            
            tts = gTTS(text=text, lang=lang_code, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_bytes = audio_buffer.read()
            
            self.audio_cache[cache_key] = audio_bytes
            
            return audio_bytes
            
        except Exception as e:
            st.error(f"TTS Error: {e}")
            return None
    
    def create_audio_autoplay_html(self, audio_bytes: bytes, autoplay: bool = True) -> str:
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        autoplay_attr = "autoplay" if autoplay else ""
        html = f"""
        <audio id="autoPlayAudio" controls {autoplay_attr} style="width: 100%; margin-top: 10px;">
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
            "speech_rate": "Normal"  # Fixed: Capitalized to match options
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
            audio_bytes = st.session_state.ap.text_to_speech(response, lang_code=tts_lang)
        
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
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("### ⚙️ Voice Settings")
        
        auto_play = st.checkbox(
            "Auto-play voice responses 🔊",
            value=st.session_state.user_preferences.get("auto_play", True),
            help="Automatically play voice responses"
        )
        st.session_state.user_preferences["auto_play"] = auto_play
        
        voice_enabled = st.checkbox(
            "Enable voice responses",
            value=st.session_state.user_preferences.get("voice_enabled", True),
            help="Enable text-to-speech"
        )
        st.session_state.user_preferences["voice_enabled"] = voice_enabled
        
        if voice_enabled:
            tts_lang = st.radio(
                "Response language",
                ["Malayalam", "English"],
                index=0 if st.session_state.user_preferences.get("tts_language") == "ml" else 1
            )
            st.session_state.user_preferences["tts_language"] = "ml" if tts_lang == "Malayalam" else "en"
            
            # Fixed: Using the correct case for default value
            speech_rate_options = ["Slow", "Normal", "Fast"]
            current_speech_rate = st.session_state.user_preferences.get("speech_rate", "Normal")
            
            # Ensure current value is in the options
            if current_speech_rate not in speech_rate_options:
                current_speech_rate = "Normal"
            
            speech_rate = st.select_slider(
                "Speech speed",
                options=speech_rate_options,
                value=current_speech_rate
            )
            st.session_state.user_preferences["speech_rate"] = speech_rate
        
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
                test_text = "Hello! This is a test of the voice assistant."
                if st.session_state.user_preferences["tts_language"] == "ml":
                    test_text = "നമസ്കാരം! ഇത് വോയ്സ് അസിസ്റ്റന്റിന്റെ ഒരു പരീക്ഷണമാണ്."
                
                audio_bytes = st.session_state.ap.text_to_speech(
                    test_text, 
                    lang_code=st.session_state.user_preferences["tts_language"]
                )
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

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
            🔊 Voice Auto-play Enabled
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        auto_play_status = "🟢 ON" if st.session_state.user_preferences["auto_play"] else "🔴 OFF"
        st.markdown(f"**Auto-play:** {auto_play_status}")
    
    with col2:
        voice_status = "🟢 ON" if st.session_state.user_preferences["voice_enabled"] else "🔴 OFF"
        st.markdown(f"**Voice:** {voice_status}")
    
    with col3:
        lang_name = "Malayalam" if st.session_state.user_preferences["tts_language"] == "ml" else "English"
        st.markdown(f"**Voice Lang:** {lang_name}")
    
    with col4:
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
                <p>Click below to record in Malayalam. The response will play automatically.</p>
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
            st.info("👋 **Welcome to സർവജ്ഞ with Auto-play Voice!**")
        
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
                
                st.markdown(f"""
                <div class="assistant-message">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>🤖 സർവജ്ഞ</strong>
                            {autoplay_indicator}
                        </div>
                        <small style="color: #666;">{message['timestamp']}</small>
                    </div>
                    <p style="margin-top: 10px;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if (message.get('audio_bytes') and 
                    st.session_state.user_preferences["voice_enabled"]):
                    
                    st.audio(message['audio_bytes'], format="audio/mp3")
                    
                    col_play, col_info = st.columns([1, 3])
                    with col_play:
                        if st.button(f"🔊 Play Again", key=f"play_{i}"):
                            st.session_state.should_autoplay = True
                            st.session_state.last_audio_data = message['audio_bytes']
                            st.rerun()
                    
                    with col_info:
                        lang_name = "Malayalam" if st.session_state.user_preferences["tts_language"] == "ml" else "English"
                        speed = st.session_state.user_preferences['speech_rate']
                        st.caption(f"Voice: {lang_name} | Speed: {speed}")
        
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
        
        st.markdown("### 🔊 Voice Test")
        test_text = st.text_input("Test text:", "This is a voice test.")
        
        col_test1, col_test2 = st.columns(2)
        with col_test1:
            if st.button("Test English", use_container_width=True):
                audio_bytes = st.session_state.ap.text_to_speech(test_text, lang_code="en")
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
        
        with col_test2:
            if st.button("Test Malayalam", use_container_width=True):
                audio_bytes = st.session_state.ap.text_to_speech(test_text, lang_code="ml")
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
    
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
        <p>🔊 <strong>Voice Assistant with Auto-play</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()