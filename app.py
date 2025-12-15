"""
സർവജ്ഞ (Sarva-jña) - LBS College AI Voice Assistant
Voice-optimized assistant with short, natural responses for Sarvam AI TTS
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
    st.error("⚠️ PPLX_API_KEY is missing!")
    st.stop()

if not SARVAM_API_KEY:
    st.error("⚠️ SARVAM_API_KEY is missing!")
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
        
        # Direct pattern matching
        for entry in self.faqs:
            for pattern in entry.get("question_patterns", []):
                p_norm = self._normalize(pattern)
                if p_norm in q_norm or q_norm in p_norm:
                    return entry
        
        # Tag-based matching
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
    
    def extract_specific_answer(self, query: str, kb_entry: Dict[str, Any]) -> Optional[str]:
        """Extract only the specific answer related to the query"""
        if not kb_entry:
            return None
        
        facts = kb_entry.get("answer_facts", {})
        q_lower = query.lower()
        
        # Keywords to fact mapping
        keyword_mapping = {
            "phone": ["phone", "contact_phone", "telephone", "mobile"],
            "email": ["email", "contact_email", "mail"],
            "address": ["address", "location", "place"],
            "time": ["timing", "hours", "time", "schedule"],
            "fee": ["fee", "fees", "cost", "price", "amount"],
            "course": ["courses", "programs", "departments"],
            "principal": ["principal", "head"],
            "website": ["website", "url", "site"],
            "admission": ["admission", "apply", "eligibility"],
            "hostel": ["hostel", "accommodation", "rooms"],
            "library": ["library", "books", "reading"],
            "placement": ["placement", "jobs", "recruitment"],
            "bus": ["bus", "transport", "route"],
            "canteen": ["canteen", "food", "mess"],
        }
        
        # Find matching keywords in query
        for keyword, fact_keys in keyword_mapping.items():
            if keyword in q_lower:
                for fact_key in fact_keys:
                    for key, value in facts.items():
                        if fact_key in key.lower():
                            return str(value)
        
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
# VOICE-OPTIMIZED AI PROCESSOR
# -------------------------------
class AIProcessor:
    def __init__(self, model: str = "sonar"):
        self.model = model
        self.client = pplx_client
    
    def generate_voice_response(self, user_query: str, kb_entry: Dict[str, Any], lang_mode: str, specific_answer: str = None) -> str:
        """Generate short, voice-friendly response"""
        
        facts = kb_entry.get("answer_facts", {})
        
        # If we have a specific answer, use it directly
        if specific_answer:
            return self._format_for_voice(user_query, specific_answer, lang_mode)
        
        # Build minimal context
        kb_text = "\n".join(f"{key}: {value}" for key, value in facts.items())
        
        if lang_mode == "en":
            system_prompt = """You are a voice assistant for LBS College. 

CRITICAL RULES:
1. Answer ONLY the specific question asked
2. Give ONE short sentence answer (max 20 words)
3. Do NOT list all information - only what was asked
4. Speak naturally like a human
5. No bullet points, numbers, or lists
6. Response must be perfect for text-to-speech

Examples:
Q: "What is the phone number?" → "The college phone number is 04994 230 008."
Q: "Where is the college?" → "LBS College is located in Kasaragod, Kerala."
Q: "Library timing?" → "The library is open from 9 AM to 5 PM on weekdays."
"""
        else:
            system_prompt = """You are a Malayalam voice assistant for LBS College.

CRITICAL RULES:
1. Answer ONLY the specific question asked
2. Give ONE short sentence answer (max 20 words)
3. Use simple Malayalam or Manglish
4. Speak naturally like talking to a friend
5. No bullet points or lists
6. Response must be perfect for text-to-speech

Examples:
Q: "Phone number എന്താണ്?" → "College ന്റെ phone number 04994 230 008 ആണ്."
Q: "College എവിടെയാണ്?" → "LBS College Kasaragod ൽ ആണ് സ്ഥിതി ചെയ്യുന്നത്."
"""
        
        user_message = f"""Question: {user_query}

Available Information:
{kb_text}

Give a SHORT, DIRECT answer to ONLY the question asked. One sentence maximum."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=80
            )
            
            answer = response.choices[0].message.content.strip()
            return self._clean_for_tts(answer)
            
        except Exception as e:
            st.error(f"AI Error: {e}")
            return self._fallback_response(specific_answer or list(facts.values())[0] if facts else "Sorry, I don't have that information.", lang_mode)
    
    def _format_for_voice(self, query: str, answer: str, lang_mode: str) -> str:
        """Format a direct answer for voice output"""
        q_lower = query.lower()
        
        if lang_mode == "en":
            if "phone" in q_lower or "number" in q_lower:
                return f"The phone number is {answer}."
            elif "email" in q_lower:
                return f"The email address is {answer}."
            elif "time" in q_lower or "hour" in q_lower:
                return f"The timing is {answer}."
            elif "address" in q_lower or "where" in q_lower or "location" in q_lower:
                return f"The location is {answer}."
            elif "fee" in q_lower or "cost" in q_lower:
                return f"The fee is {answer}."
            else:
                return f"It is {answer}."
        else:
            if "phone" in q_lower or "number" in q_lower or "നമ്പർ" in query:
                return f"Phone number {answer} ആണ്."
            elif "email" in q_lower:
                return f"Email {answer} ആണ്."
            elif "time" in q_lower or "സമയം" in query:
                return f"സമയം {answer} ആണ്."
            elif "എവിടെ" in query or "location" in q_lower:
                return f"സ്ഥലം {answer} ആണ്."
            else:
                return f"{answer}."
    
    def _clean_for_tts(self, text: str) -> str:
        """Clean text for TTS output"""
        # Remove markdown formatting
        text = text.replace("**", "").replace("*", "")
        text = text.replace("#", "").replace("`", "")
        
        # Remove bullet points and numbers at start
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                # Remove leading bullets/numbers
                if line.startswith(("-", "•", "●")):
                    line = line[1:].strip()
                if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
                    line = line[2:].strip()
                cleaned_lines.append(line)
        
        text = " ".join(cleaned_lines)
        
        # Clean extra spaces
        while "  " in text:
            text = text.replace("  ", " ")
        
        return text.strip()
    
    def _fallback_response(self, answer: str, lang_mode: str) -> str:
        """Generate fallback response"""
        if lang_mode == "en":
            return f"Here's what I found: {answer}"
        else:
            return f"ഇതാണ് വിവരം: {answer}"
    
    def generate_not_found_response(self, lang_mode: str) -> str:
        """Generate voice-friendly not found response"""
        if lang_mode == "en":
            return "Sorry, I don't have that information. Please ask something else about the college."
        else:
            return "ക്ഷമിക്കണം, ആ വിവരം എന്റെ കയ്യിൽ ഇല്ല. കോളേജിനെ കുറിച്ച് മറ്റെന്തെങ്കിലും ചോദിക്കൂ."

# -------------------------------
# SARVAM AI TTS PROCESSOR
# -------------------------------
class AudioProcessor:
    def __init__(self):
        self.client = sarvam_client
        self.language_codes = {
            "en": "en-IN",
            "ml": "ml-IN",
            "hi": "hi-IN",
            "ta": "ta-IN",
            "te": "te-IN",
            "kn": "kn-IN",
        }
        self.audio_cache = {}
    
    def get_pace_value(self, speech_rate: str) -> float:
        pace_map = {"Slow": 0.85, "Normal": 1.0, "Fast": 1.15}
        return pace_map.get(speech_rate, 1.0)
    
    def text_to_speech(
        self, 
        text: str, 
        lang_code: str = "ml",
        speaker: str = "arya",
        pitch: float = 0,
        pace: float = 1.0,
        loudness: float = 1.5,
        sample_rate: int = 22050
    ) -> Optional[bytes]:
        """Convert text to speech using Sarvam AI"""
        try:
            if not text or not text.strip():
                return None
            
            # Clean text for TTS
            text = self._prepare_text_for_tts(text)
            
            target_language = self.language_codes.get(lang_code, "en-IN")
            cache_key = f"{hash(text)}_{target_language}_{speaker}_{pace}"
            
            if cache_key in self.audio_cache:
                return self.audio_cache[cache_key]
            
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
            
            audio_bytes = self._extract_audio(response)
            
            if audio_bytes:
                self.audio_cache[cache_key] = audio_bytes
            
            return audio_bytes
            
        except Exception as e:
            st.error(f"TTS Error: {e}")
            return None
    
    def _prepare_text_for_tts(self, text: str) -> str:
        """Prepare text for optimal TTS output"""
        # Remove special characters that don't speak well
        text = text.replace("@", " at ")
        text = text.replace("&", " and ")
        text = text.replace("%", " percent ")
        text = text.replace("+", " plus ")
        text = text.replace("=", " equals ")
        
        # Add pauses with commas
        text = text.replace(" - ", ", ")
        
        # Clean multiple spaces
        while "  " in text:
            text = text.replace("  ", " ")
        
        return text.strip()
    
    def _extract_audio(self, response) -> Optional[bytes]:
        """Extract audio bytes from Sarvam AI response"""
        try:
            if hasattr(response, 'audios') and response.audios:
                audio_data = response.audios[0]
                if isinstance(audio_data, str):
                    return base64.b64decode(audio_data)
                return audio_data
            elif hasattr(response, 'audio'):
                if isinstance(response.audio, str):
                    return base64.b64decode(response.audio)
                return response.audio
            elif isinstance(response, dict):
                if 'audios' in response and response['audios']:
                    return base64.b64decode(response['audios'][0])
                if 'audio' in response:
                    return base64.b64decode(response['audio'])
            elif isinstance(response, str):
                return base64.b64decode(response)
            elif isinstance(response, bytes):
                return response
        except Exception as e:
            st.error(f"Audio extraction error: {e}")
        return None
    
    def create_autoplay_html(self, audio_bytes: bytes) -> str:
        """Create HTML for audio autoplay"""
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return f"""
        <audio id="voiceResponse" autoplay controls style="width:100%; margin-top:10px;">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
        <script>
        (function() {{
            var audio = document.getElementById('voiceResponse');
            if(audio) {{
                audio.play().catch(function(e) {{
                    console.log('Autoplay blocked:', e);
                }});
            }}
        }})();
        </script>
        """

# -------------------------------
# SESSION STATE
# -------------------------------
def init_session():
    defaults = {
        "messages": [],
        "preferences": {
            "voice_enabled": True,
            "auto_play": True,
            "tts_language": "ml",
            "speaker": "arya",
            "speech_rate": "Normal",
            "pitch": 0,
            "loudness": 1.5
        },
        "kb": None,
        "lh": None,
        "ai": None,
        "ap": None,
        "last_audio": None,
        "autoplay_pending": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if st.session_state.kb is None:
        st.session_state.kb = KnowledgeBase("faq_data.json")
    if st.session_state.lh is None:
        st.session_state.lh = LanguageHandler()
    if st.session_state.ai is None:
        st.session_state.ai = AIProcessor()
    if st.session_state.ap is None:
        st.session_state.ap = AudioProcessor()

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="സർവജ്ഞ – Voice Assistant",
    page_icon="🎤",
    layout="wide"
)

st.markdown("""
<style>
.header {
    background: linear-gradient(135deg, #1a5276 0%, #2e86ab 100%);
    color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 25px;
}
.user-msg {
    background: #e8f5e9;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #4caf50;
    margin: 10px 0;
}
.bot-msg {
    background: #e3f2fd;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #2196f3;
    margin: 10px 0;
}
.voice-box {
    background: #fff3e0;
    padding: 20px;
    border-radius: 10px;
    border: 2px dashed #ff9800;
    text-align: center;
    margin: 15px 0;
}
.quick-btn {
    margin: 3px 0;
}
</style>
""", unsafe_allow_html=True)

init_session()

# -------------------------------
# CORE FUNCTIONS
# -------------------------------
def process_query(query: str, is_voice: bool = False):
    """Process user query and generate voice response"""
    
    # Detect language
    lang_mode = st.session_state.lh.detect_language_mode(query)
    
    # Convert Malayalam script to Manglish for processing
    if lang_mode == "ml_script":
        processed_query = st.session_state.lh.malayalam_to_manglish(query)
    else:
        processed_query = query
    
    # Search knowledge base
    kb_entry = st.session_state.kb.get_relevant_info(processed_query)
    
    if kb_entry:
        # Try to get specific answer first
        specific_answer = st.session_state.kb.extract_specific_answer(processed_query, kb_entry)
        
        # Generate voice-optimized response
        response = st.session_state.ai.generate_voice_response(
            processed_query, 
            kb_entry, 
            lang_mode,
            specific_answer
        )
    else:
        response = st.session_state.ai.generate_not_found_response(lang_mode)
    
    # Generate audio
    audio_bytes = None
    if st.session_state.preferences["voice_enabled"]:
        prefs = st.session_state.preferences
        audio_bytes = st.session_state.ap.text_to_speech(
            text=response,
            lang_code=prefs["tts_language"],
            speaker=prefs["speaker"],
            pitch=prefs["pitch"],
            pace=st.session_state.ap.get_pace_value(prefs["speech_rate"]),
            loudness=prefs["loudness"]
        )
    
    # Save messages
    timestamp = datetime.now().strftime("%H:%M")
    
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "time": timestamp,
        "is_voice": is_voice
    })
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "time": timestamp,
        "audio": audio_bytes
    })
    
    if audio_bytes and st.session_state.preferences["auto_play"]:
        st.session_state.last_audio = audio_bytes
        st.session_state.autoplay_pending = True

def clear_chat():
    st.session_state.messages = []
    st.session_state.last_audio = None
    st.session_state.autoplay_pending = False

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.markdown("## 🎤 സർവജ്ഞ")
    st.caption("LBS College Voice Assistant")
    
    st.divider()
    
    st.markdown("### 🔊 Voice Settings")
    
    st.session_state.preferences["auto_play"] = st.checkbox(
        "Auto-play responses",
        value=st.session_state.preferences["auto_play"]
    )
    
    st.session_state.preferences["voice_enabled"] = st.checkbox(
        "Enable voice",
        value=st.session_state.preferences["voice_enabled"]
    )
    
    if st.session_state.preferences["voice_enabled"]:
        lang = st.radio(
            "Language",
            ["Malayalam", "English"],
            index=0 if st.session_state.preferences["tts_language"] == "ml" else 1
        )
        st.session_state.preferences["tts_language"] = "ml" if lang == "Malayalam" else "en"
        
        st.session_state.preferences["speaker"] = st.selectbox(
            "Voice",
            ["arya", "meera", "pavithra", "maitreyi"],
            index=["arya", "meera", "pavithra", "maitreyi"].index(
                st.session_state.preferences["speaker"]
            )
        )
        
        st.session_state.preferences["speech_rate"] = st.select_slider(
            "Speed",
            ["Slow", "Normal", "Fast"],
            value=st.session_state.preferences["speech_rate"]
        )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        clear_chat()
        st.rerun()
    
    if st.button("🔊 Test Voice", use_container_width=True):
        test = "നമസ്കാരം, ഞാൻ സർവജ്ഞ" if st.session_state.preferences["tts_language"] == "ml" else "Hello, I am Sarvajna"
        audio = st.session_state.ap.text_to_speech(
            test,
            st.session_state.preferences["tts_language"],
            st.session_state.preferences["speaker"]
        )
        if audio:
            st.audio(audio, format="audio/wav")

# -------------------------------
# MAIN CONTENT
# -------------------------------
st.markdown("""
<div class="header">
    <h1>🎤 സർവജ്ഞ</h1>
    <p>LBS College Voice Assistant | Ask in Malayalam or English</p>
</div>
""", unsafe_allow_html=True)

# Status bar
col1, col2, col3 = st.columns(3)
with col1:
    status = "🟢" if st.session_state.preferences["voice_enabled"] else "🔴"
    st.markdown(f"**Voice:** {status}")
with col2:
    lang = "മലയാളം" if st.session_state.preferences["tts_language"] == "ml" else "English"
    st.markdown(f"**Language:** {lang}")
with col3:
    st.markdown(f"**Speaker:** {st.session_state.preferences['speaker'].title()}")

st.divider()

# Main layout
main_col, side_col = st.columns([3, 1])

with main_col:
    # Input tabs
    tab1, tab2 = st.tabs(["💬 Type", "🎤 Speak"])
    
    with tab1:
        user_input = st.text_input(
            "Ask your question:",
            placeholder="e.g., What is the college phone number?",
            key="text_input"
        )
        
        if st.button("📤 Ask", type="primary", use_container_width=True):
            if user_input.strip():
                process_query(user_input.strip())
                st.rerun()
    
    with tab2:
        st.markdown("""
        <div class="voice-box">
            <h4>🎙️ Speak in Malayalam</h4>
            <p>Click to record your question</p>
        </div>
        """, unsafe_allow_html=True)
        
        voice_text = speech_to_text(
            language='ml-IN',
            start_prompt="🎙️ Record",
            stop_prompt="⏹️ Stop",
            just_once=True,
            use_container_width=True,
            key="voice_input"
        )
        
        if voice_text:
            st.info(f"🎤 Heard: {voice_text}")
            process_query(voice_text, is_voice=True)
            st.rerun()
    
    st.divider()
    
    # Conversation
    st.markdown("### 💬 Conversation")
    
    if not st.session_state.messages:
        st.info("👋 Ask me anything about LBS College!")
    
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            icon = "🎤" if msg.get("is_voice") else "💬"
            st.markdown(f"""
            <div class="user-msg">
                <strong>You {icon}</strong> <small>({msg['time']})</small><br>
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-msg">
                <strong>🤖 സർവജ്ഞ</strong> <small>({msg['time']})</small><br>
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)
            
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/wav")
    
    # Autoplay latest audio
    if st.session_state.autoplay_pending and st.session_state.last_audio:
        st.session_state.autoplay_pending = False
        html = st.session_state.ap.create_autoplay_html(st.session_state.last_audio)
        st.markdown(html, unsafe_allow_html=True)

with side_col:
    st.markdown("### ⚡ Quick Ask")
    
    quick_questions = [
        ("📞 Phone number?", "What is the phone number?"),
        ("📍 Location?", "Where is the college located?"),
        ("📧 Email?", "What is the email address?"),
        ("🕐 Library timing?", "What is the library timing?"),
        ("📚 Courses?", "What courses are available?"),
        ("🏠 Hostel info?", "Tell me about hostel facilities"),
        ("💼 Placements?", "What about placements?"),
        ("📝 Admission?", "How to apply for admission?"),
    ]
    
    for label, question in quick_questions:
        if st.button(label, key=f"q_{label}", use_container_width=True):
            process_query(question)
            st.rerun()
    
    st.divider()
    
    st.markdown("### 🗣️ Malayalam")
    
    ml_questions = [
        ("ഫോൺ നമ്പർ?", "ഫോൺ നമ്പർ എന്താണ്?"),
        ("എവിടെയാണ്?", "കോളേജ് എവിടെയാണ്?"),
        ("സമയം?", "ലൈബ്രറി സമയം എന്താണ്?"),
    ]
    
    for label, question in ml_questions:
        if st.button(label, key=f"ml_{label}", use_container_width=True):
            process_query(question)
            st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align:center; color:#666; padding:15px;">
    <p>🎓 <strong>LBS College of Engineering, Kasaragod</strong></p>
    <p>🔊 Powered by <strong>Sarvam AI</strong></p>
</div>
""", unsafe_allow_html=True)
