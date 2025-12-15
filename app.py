"""
സർവജ്ഞ (Sarva-jña) - LBS College AI Voice Assistant
Smart JSON-based answers with human-like conversational responses
"""

import os
import json
import difflib
import random
import re
import base64
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

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
# CONVERSATION HANDLER
# -------------------------------
class ConversationHandler:
    """Handle greetings, small talk, and friendly interactions"""
    
    def __init__(self):
        self.bot_name = "സർവജ്ഞ"
        self.bot_name_en = "Sarvajna"
        
        # Greeting patterns with responses
        self.greeting_patterns = {
            "sugamano": ["സുഖമാണ്! നിങ്ങൾക്കോ? എന്തെങ്കിലും സഹായം വേണോ?", "നന്നായിരിക്കുന്നു! എന്താണ് അറിയേണ്ടത്?"],
            "sukhamano": ["സുഖമാണ്! നിങ്ങൾക്കോ?", "നന്നായിരിക്കുന്നു! നിങ്ങളെ കാണുന്നതിൽ സന്തോഷം."],
            "സുഖമാണോ": ["സുഖമാണ്! നിങ്ങൾക്കോ?", "എനിക്ക് നന്നായിരിക്കുന്നു!"],
            "നമസ്കാരം": ["നമസ്കാരം! ഞാൻ സർവജ്ഞ ആണ്. എന്താണ് സഹായം വേണ്ടത്?", "നമസ്കാരം! എന്തൊക്കെ ഉണ്ട് വിശേഷം?"],
            "namaskaram": ["നമസ്കാരം! ഞാൻ സർവജ്ഞ. എന്താണ് അറിയേണ്ടത്?", "നമസ്കാരം! സഹായിക്കാൻ തയ്യാറാണ്!"],
            "hello": ["Hello! I'm Sarvajna. How can I help you today?", "Hi there! What would you like to know?"],
            "hi": ["Hi! I'm Sarvajna. How can I assist you?", "Hello! What can I help with?"],
            "hey": ["Hey! What's up? How can I help?", "Hey there! I'm here to help."],
            "good morning": ["Good morning! Hope you're having a great day. How can I help?", "Good morning! What do you need?"],
            "good afternoon": ["Good afternoon! How can I assist you today?", "Good afternoon! What would you like to know?"],
            "good evening": ["Good evening! How may I help you?", "Good evening! What can I do for you?"],
        }
        
        # How are you patterns
        self.how_are_you_patterns = {
            "how are you": ["I'm doing great, thank you! How about you?", "I'm wonderful! Ready to help you."],
            "how r u": ["I'm doing great! What do you need help with?", "All good here! How can I assist?"],
            "what's up": ["Not much, just here to help! What do you need?", "All good! Ready to answer your questions."],
            "enthokke und": ["എല്ലാം നന്നായിരിക്കുന്നു! നിങ്ങൾക്കോ?", "സുഖമാണ്!"],
            "എന്തൊക്കെ ഉണ്ട്": ["എല്ലാം നന്നായിരിക്കുന്നു!", "നന്നായിട്ടുണ്ട്!"],
        }
        
        # Thank you patterns
        self.thank_you_patterns = {
            "thank you": ["You're welcome! Feel free to ask anything else.", "My pleasure! Is there anything else?"],
            "thanks": ["You're welcome!", "No problem at all!", "Glad I could help!"],
            "nanni": ["സ്വാഗതം! വേറെ എന്തെങ്കിലും വേണോ?", "സന്തോഷം!"],
            "നന്ദി": ["സ്വാഗതം!", "സന്തോഷമായി!"],
        }
        
        # Goodbye patterns
        self.goodbye_patterns = {
            "bye": ["Goodbye! Have a great day!", "Bye! Come back anytime!"],
            "goodbye": ["Goodbye! Take care!", "See you soon!"],
            "pinne kanam": ["പിന്നെ കാണാം! നല്ല ദിവസം!", "ശരി, പിന്നെ കാണാം!"],
            "പിന്നെ കാണാം": ["ശരി, പിന്നെ കാണാം!", "പിന്നെ കാണാം!"],
        }
        
        # About me patterns
        self.about_me_patterns = {
            "who are you": ["I'm Sarvajna, your AI assistant for LBS College. I can help with college info!", 
                           "I'm Sarvajna! I help with LBS College information."],
            "what is your name": ["My name is Sarvajna! I'm the LBS College assistant.", 
                                  "I'm called Sarvajna!"],
            "nee aara": ["ഞാൻ സർവജ്ഞ ആണ്! LBS കോളേജിന്റെ AI അസിസ്റ്റന്റ്.", 
                        "എന്റെ പേര് സർവജ്ഞ."],
            "നീ ആരാ": ["ഞാൻ സർവജ്ഞ!", "എന്റെ പേര് സർവജ്ഞ."],
        }
    
    def get_time_based_greeting(self, lang: str = "en") -> str:
        hour = datetime.now().hour
        if lang == "ml":
            if 5 <= hour < 12:
                return random.choice(["സുപ്രഭാതം!", "ഗുഡ് മോർണിംഗ്!"])
            elif 12 <= hour < 17:
                return random.choice(["ശുഭ ഉച്ച!", "ഗുഡ് ആഫ്റ്റർനൂൺ!"])
            elif 17 <= hour < 21:
                return random.choice(["ശുഭ സന്ധ്യ!", "ഗുഡ് ഈവനിംഗ്!"])
            else:
                return "ശുഭ രാത്രി!"
        else:
            if 5 <= hour < 12:
                return "Good morning!"
            elif 12 <= hour < 17:
                return "Good afternoon!"
            elif 17 <= hour < 21:
                return "Good evening!"
            else:
                return "Hello!"
    
    def get_welcome_message(self, lang: str = "en") -> str:
        time_greeting = self.get_time_based_greeting(lang)
        if lang == "ml":
            return f"{time_greeting} ഞാൻ സർവജ്ഞ ആണ്, LBS കോളേജിന്റെ AI അസിസ്റ്റന്റ്. കോളേജിനെ കുറിച്ച് എന്തും ചോദിക്കാം!"
        else:
            return f"{time_greeting} I'm Sarvajna, the AI assistant for LBS College. Feel free to ask me anything about the college!"
    
    def is_conversation_query(self, query: str) -> Tuple[bool, str, str]:
        query_lower = query.lower().strip()
        query_clean = ''.join(c for c in query_lower if c.isalnum() or c.isspace() or ord(c) > 127)
        
        all_patterns = [
            (self.greeting_patterns, "greeting"),
            (self.how_are_you_patterns, "how_are_you"),
            (self.thank_you_patterns, "thank_you"),
            (self.goodbye_patterns, "goodbye"),
            (self.about_me_patterns, "about_me"),
        ]
        
        for patterns, pattern_type in all_patterns:
            for key, responses in patterns.items():
                if key in query_clean or query_clean in key:
                    return True, random.choice(responses), pattern_type
        
        return False, "", ""


# -------------------------------
# SMART KNOWLEDGE BASE CLASS
# -------------------------------
class KnowledgeBase:
    """Smart knowledge base with intelligent query matching"""
    
    def __init__(self, file_name: str = "faq_data.json"):
        self.file_path = file_name
        self.faqs = []
        self.load_faqs()
        
        # Question type mappings for smart extraction
        self.question_keywords = {
            # Contact related
            "phone": ["phone", "call", "contact number", "telephone", "mobile", "number", "ഫോൺ", "നമ്പർ", "വിളിക്കാൻ"],
            "email": ["email", "mail", "e-mail", "ഇമെയിൽ", "മെയിൽ"],
            "address": ["address", "location", "where", "place", "situated", "എവിടെ", "സ്ഥലം", "അഡ്രസ്"],
            "website": ["website", "site", "url", "web", "online", "വെബ്സൈറ്റ്"],
            
            # Timing related
            "timing": ["timing", "time", "hours", "when", "open", "close", "schedule", "സമയം", "എപ്പോൾ"],
            "working_hours": ["working hours", "office hours", "office time"],
            
            # Fee related
            "fee": ["fee", "fees", "cost", "price", "amount", "charge", "ഫീസ്", "പണം", "തുക"],
            "tuition": ["tuition", "tuition fee"],
            "hostel_fee": ["hostel fee", "hostel charge", "accommodation fee"],
            
            # Course related
            "courses": ["course", "courses", "program", "programmes", "branch", "department", "stream", "കോഴ്സ്", "ബ്രാഞ്ച്"],
            "seats": ["seats", "intake", "capacity", "how many students"],
            "duration": ["duration", "years", "how long", "period"],
            
            # Admission related
            "admission": ["admission", "apply", "join", "enroll", "registration", "അഡ്മിഷൻ", "ചേരാൻ"],
            "eligibility": ["eligibility", "qualification", "requirement", "criteria", "who can apply"],
            "documents": ["documents", "papers", "certificates", "required documents"],
            
            # Facility related
            "hostel": ["hostel", "accommodation", "staying", "room", "ഹോസ്റ്റൽ", "താമസം"],
            "library": ["library", "books", "reading room", "ലൈബ്രറി", "പുസ്തകം"],
            "canteen": ["canteen", "food", "mess", "cafeteria", "ക്യാന്റീൻ", "ഭക്ഷണം"],
            "lab": ["lab", "laboratory", "practical", "ലാബ്"],
            "sports": ["sports", "games", "playground", "ground", "കളി", "സ്പോർട്സ്"],
            "wifi": ["wifi", "internet", "network", "വൈഫൈ"],
            
            # Placement related
            "placement": ["placement", "job", "recruit", "company", "career", "പ്ലേസ്മെന്റ്", "ജോലി"],
            "salary": ["salary", "package", "ctc", "offer", "ശമ്പളം"],
            "companies": ["companies", "recruiters", "which companies"],
            
            # Transport related
            "bus": ["bus", "transport", "route", "vehicle", "ബസ്"],
            
            # People related
            "principal": ["principal", "head", "director", "പ്രിൻസിപ്പൽ"],
            "faculty": ["faculty", "teachers", "professors", "staff", "ടീച്ചർ", "അധ്യാപകർ"],
            
            # General
            "about": ["about", "tell me about", "what is", "info", "information", "എന്താണ്", "കുറിച്ച്"],
            "facilities": ["facilities", "amenities", "what facilities", "സൗകര്യങ്ങൾ"],
        }
    
    def load_faqs(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.faqs = json.load(f)
            elif os.path.exists(f"data/{self.file_path}"):
                with open(f"data/{self.file_path}", "r", encoding="utf-8") as f:
                    self.faqs = json.load(f)
            else:
                # Create sample FAQ data if not exists
                self.faqs = self._create_sample_data()
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            self.faqs = self._create_sample_data()
    
    def _create_sample_data(self) -> List[Dict]:
        """Create sample FAQ data for demonstration"""
        return [
            {
                "id": "college_contact",
                "question_patterns": ["contact", "phone number", "how to contact", "call"],
                "tags": ["contact", "phone", "call", "reach"],
                "answer_facts": {
                    "phone": "04994 230 008",
                    "email": "principal@lbscek.ac.in",
                    "address": "LBS College of Engineering, Kasaragod, Kerala - 671542",
                    "website": "www.lbscek.ac.in"
                }
            },
            {
                "id": "college_timing",
                "question_patterns": ["timing", "working hours", "college time", "when open"],
                "tags": ["timing", "hours", "schedule", "time"],
                "answer_facts": {
                    "college_timing": "9:00 AM to 4:30 PM",
                    "office_timing": "10:00 AM to 5:00 PM",
                    "working_days": "Monday to Friday",
                    "library_timing": "9:00 AM to 6:00 PM"
                }
            },
            {
                "id": "courses",
                "question_patterns": ["courses", "branches", "programs", "what courses"],
                "tags": ["course", "branch", "program", "department", "study"],
                "answer_facts": {
                    "courses": "Computer Science, Electronics, Electrical, Mechanical, Civil Engineering",
                    "total_courses": "5 B.Tech programs",
                    "duration": "4 years",
                    "intake_per_branch": "60 students per branch"
                }
            },
            {
                "id": "admission",
                "question_patterns": ["admission", "how to apply", "join college", "admission process"],
                "tags": ["admission", "apply", "join", "enroll"],
                "answer_facts": {
                    "admission_process": "Through KEAM counselling",
                    "eligibility": "Plus Two with Physics, Chemistry, and Mathematics",
                    "minimum_marks": "50% in PCM for General, 45% for Reserved",
                    "admission_period": "June to August"
                }
            },
            {
                "id": "fees",
                "question_patterns": ["fees", "fee structure", "cost", "how much"],
                "tags": ["fee", "cost", "payment", "tuition"],
                "answer_facts": {
                    "tuition_fee": "₹35,000 per year (Government quota)",
                    "hostel_fee": "₹15,000 per year",
                    "exam_fee": "₹1,500 per semester",
                    "caution_deposit": "₹5,000 (refundable)"
                }
            },
            {
                "id": "hostel",
                "question_patterns": ["hostel", "accommodation", "stay", "rooms"],
                "tags": ["hostel", "accommodation", "room", "stay", "living"],
                "answer_facts": {
                    "hostel_availability": "Separate hostels for boys and girls",
                    "room_type": "Shared rooms with 2-3 students",
                    "facilities": "WiFi, mess, recreation room, 24/7 security",
                    "hostel_fee": "₹15,000 per year"
                }
            },
            {
                "id": "placement",
                "question_patterns": ["placement", "job", "companies", "recruitment"],
                "tags": ["placement", "job", "career", "company", "recruit"],
                "answer_facts": {
                    "placement_rate": "85% average placement rate",
                    "top_companies": "TCS, Infosys, Wipro, UST Global, IBS Software",
                    "highest_package": "₹12 LPA",
                    "average_package": "₹4.5 LPA"
                }
            },
            {
                "id": "library",
                "question_patterns": ["library", "books", "reading"],
                "tags": ["library", "book", "reading", "study"],
                "answer_facts": {
                    "library_timing": "9:00 AM to 6:00 PM",
                    "total_books": "Over 25,000 books",
                    "digital_library": "Access to IEEE, Springer, and other journals",
                    "seating_capacity": "200 students"
                }
            },
            {
                "id": "principal",
                "question_patterns": ["principal", "head", "director"],
                "tags": ["principal", "head", "management"],
                "answer_facts": {
                    "principal_name": "Dr. K. Radhakrishnan",
                    "designation": "Principal",
                    "contact": "principal@lbscek.ac.in"
                }
            },
            {
                "id": "location",
                "question_patterns": ["where", "location", "address", "situated"],
                "tags": ["location", "address", "place", "where"],
                "answer_facts": {
                    "full_address": "LBS College of Engineering, Povval, Kasaragod, Kerala - 671542",
                    "district": "Kasaragod",
                    "state": "Kerala",
                    "nearest_railway": "Kasaragod Railway Station (8 km)",
                    "nearest_airport": "Mangalore International Airport (55 km)"
                }
            }
        ]
    
    def _normalize(self, text: str) -> str:
        return text.lower().strip()
    
    def get_question_type(self, query: str) -> List[str]:
        """Identify what type of information the user is asking for"""
        query_lower = query.lower()
        detected_types = []
        
        for q_type, keywords in self.question_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected_types.append(q_type)
                    break
        
        return detected_types if detected_types else ["general"]
    
    def get_relevant_info(self, query: str) -> Optional[Dict[str, Any]]:
        """Find the most relevant FAQ entry for the query"""
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
        
        # Check for tag matches in query
        for tag in all_tags:
            if tag in q_norm:
                return tag_to_entry[tag]
        
        # Fuzzy matching
        words = q_norm.split()
        for word in words:
            if len(word) > 3:
                close_matches = difflib.get_close_matches(word, all_tags, n=1, cutoff=0.6)
                if close_matches:
                    return tag_to_entry[close_matches[0]]
        
        return None
    
    def extract_specific_answer(self, query: str, kb_entry: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """Extract only the specific answer related to the query
        Returns: (specific_value, fact_key)
        """
        if not kb_entry:
            return None, ""
        
        facts = kb_entry.get("answer_facts", {})
        query_lower = query.lower()
        
        # Get question types
        question_types = self.get_question_type(query)
        
        # Map question types to fact keys
        type_to_fact_keys = {
            "phone": ["phone", "contact_phone", "telephone", "mobile"],
            "email": ["email", "contact_email", "mail"],
            "address": ["address", "full_address", "location"],
            "website": ["website", "url", "site"],
            "timing": ["timing", "college_timing", "office_timing", "hours", "library_timing"],
            "fee": ["fee", "tuition_fee", "total_fee", "fees"],
            "hostel_fee": ["hostel_fee", "hostel_charge"],
            "courses": ["courses", "programs", "branches"],
            "seats": ["seats", "intake", "intake_per_branch"],
            "duration": ["duration", "years"],
            "admission": ["admission_process", "how_to_apply"],
            "eligibility": ["eligibility", "requirement", "minimum_marks"],
            "hostel": ["hostel_availability", "facilities", "room_type"],
            "library": ["library_timing", "total_books", "seating_capacity"],
            "placement": ["placement_rate", "companies", "top_companies"],
            "salary": ["highest_package", "average_package", "package"],
            "principal": ["principal_name", "name"],
            "about": list(facts.keys())[:3] if facts else [],  # First 3 facts for general
        }
        
        # Find matching fact
        for q_type in question_types:
            if q_type in type_to_fact_keys:
                for fact_key in type_to_fact_keys[q_type]:
                    for key, value in facts.items():
                        if fact_key in key.lower() or key.lower() in fact_key:
                            return str(value), key
        
        # If no specific match, check for keywords directly in fact keys
        for key, value in facts.items():
            key_lower = key.lower().replace("_", " ")
            for word in query_lower.split():
                if len(word) > 3 and word in key_lower:
                    return str(value), key
        
        return None, ""


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
# HUMAN-LIKE RESPONSE GENERATOR
# -------------------------------
class HumanResponseGenerator:
    """Generate natural, human-like responses based on JSON data"""
    
    def __init__(self):
        # Response templates for different question types
        self.templates_en = {
            "phone": [
                "The phone number is {value}. Feel free to call!",
                "You can reach us at {value}.",
                "Our contact number is {value}.",
            ],
            "email": [
                "The email address is {value}.",
                "You can email us at {value}.",
                "Drop us a mail at {value}.",
            ],
            "address": [
                "We're located at {value}.",
                "The college is at {value}.",
                "You'll find us at {value}.",
            ],
            "timing": [
                "The timing is {value}.",
                "We're open from {value}.",
                "The hours are {value}.",
            ],
            "fee": [
                "The fee is {value}.",
                "It costs {value}.",
                "You'll need to pay {value}.",
            ],
            "courses": [
                "We offer {value}.",
                "The available courses are {value}.",
                "You can study {value} here.",
            ],
            "placement": [
                "Regarding placements, {value}.",
                "For placements, {value}.",
                "Our placement record shows {value}.",
            ],
            "hostel": [
                "About hostel, {value}.",
                "For accommodation, {value}.",
                "Hostel facility: {value}.",
            ],
            "admission": [
                "For admission, {value}.",
                "The admission process is {value}.",
                "To join, {value}.",
            ],
            "library": [
                "The library {value}.",
                "Our library has {value}.",
                "About the library, {value}.",
            ],
            "principal": [
                "Our principal is {value}.",
                "The principal's name is {value}.",
                "{value} is our principal.",
            ],
            "general": [
                "Here's what I found: {value}.",
                "{value}.",
                "The answer is {value}.",
            ],
        }
        
        self.templates_ml = {
            "phone": [
                "Phone number {value} ആണ്. വിളിക്കാം!",
                "ഞങ്ങളെ {value} ൽ ബന്ധപ്പെടാം.",
            ],
            "email": [
                "Email {value} ആണ്.",
                "{value} എന്ന email ൽ ബന്ധപ്പെടാം.",
            ],
            "address": [
                "കോളേജ് {value} ൽ ആണ്.",
                "ഞങ്ങൾ {value} ൽ ആണ്.",
            ],
            "timing": [
                "സമയം {value} ആണ്.",
                "{value} ആണ് സമയം.",
            ],
            "fee": [
                "ഫീസ് {value} ആണ്.",
                "{value} ആണ് ഫീസ്.",
            ],
            "courses": [
                "{value} ഇവിടെ പഠിക്കാം.",
                "ലഭ്യമായ courses: {value}.",
            ],
            "general": [
                "{value}.",
                "ഉത്തരം {value} ആണ്.",
            ],
        }
        
        # Friendly additions
        self.friendly_additions_en = [
            " Is there anything else you'd like to know?",
            " Feel free to ask more!",
            " Happy to help!",
            " Let me know if you need more info.",
            "",
            "",
        ]
        
        self.friendly_additions_ml = [
            " വേറെ എന്തെങ്കിലും വേണോ?",
            " കൂടുതൽ ചോദിക്കാം!",
            "",
            "",
        ]
    
    def get_question_category(self, query: str, fact_key: str) -> str:
        """Determine the category of question for template selection"""
        query_lower = query.lower()
        fact_key_lower = fact_key.lower()
        
        categories = {
            "phone": ["phone", "call", "number", "contact", "ഫോൺ", "നമ്പർ"],
            "email": ["email", "mail", "ഇമെയിൽ"],
            "address": ["address", "where", "location", "എവിടെ", "സ്ഥലം"],
            "timing": ["time", "timing", "hour", "when", "open", "സമയം"],
            "fee": ["fee", "cost", "price", "pay", "ഫീസ്"],
            "courses": ["course", "branch", "program", "study", "കോഴ്സ്"],
            "placement": ["placement", "job", "company", "salary", "package", "പ്ലേസ്മെന്റ്"],
            "hostel": ["hostel", "stay", "room", "accommodation", "ഹോസ്റ്റൽ"],
            "admission": ["admission", "join", "apply", "അഡ്മിഷൻ"],
            "library": ["library", "book", "ലൈബ്രറി"],
            "principal": ["principal", "head", "പ്രിൻസിപ്പൽ"],
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in query_lower or keyword in fact_key_lower:
                    return category
        
        return "general"
    
    def generate_response(self, query: str, value: str, fact_key: str, lang_mode: str) -> str:
        """Generate a human-like response"""
        category = self.get_question_category(query, fact_key)
        
        if lang_mode in ["ml_script", "manglish"]:
            templates = self.templates_ml.get(category, self.templates_ml["general"])
            additions = self.friendly_additions_ml
        else:
            templates = self.templates_en.get(category, self.templates_en["general"])
            additions = self.friendly_additions_en
        
        # Select random template
        template = random.choice(templates)
        response = template.format(value=value)
        
        # Add friendly ending (50% chance)
        if random.random() > 0.5:
            response += random.choice(additions)
        
        return response.strip()
    
    def generate_multi_fact_response(self, query: str, facts: Dict[str, Any], lang_mode: str) -> str:
        """Generate response when multiple facts are relevant"""
        if lang_mode in ["ml_script", "manglish"]:
            intro = random.choice(["ഇതാ വിവരങ്ങൾ:", "ഇതാണ് details:"])
        else:
            intro = random.choice(["Here's what I found:", "Here are the details:"])
        
        # Format facts naturally
        fact_strings = []
        for key, value in list(facts.items())[:3]:  # Limit to 3 facts
            clean_key = key.replace("_", " ").title()
            fact_strings.append(f"{clean_key}: {value}")
        
        return intro + " " + ". ".join(fact_strings) + "."


# -------------------------------
# VOICE-OPTIMIZED AI PROCESSOR
# -------------------------------
class AIProcessor:
    def __init__(self, model: str = "sonar"):
        self.model = model
        self.client = pplx_client
        self.response_generator = HumanResponseGenerator()
    
    def generate_voice_response(self, user_query: str, kb_entry: Dict[str, Any], lang_mode: str, specific_answer: Tuple[str, str] = None) -> str:
        """Generate short, voice-friendly response with personality"""
        
        value, fact_key = specific_answer if specific_answer else (None, "")
        facts = kb_entry.get("answer_facts", {})
        
        # If we have a specific answer, use human response generator
        if value:
            return self.response_generator.generate_response(user_query, value, fact_key, lang_mode)
        
        # If we have facts but no specific match, generate multi-fact response
        if facts:
            return self.response_generator.generate_multi_fact_response(user_query, facts, lang_mode)
        
        # Fallback to AI generation
        return self._generate_ai_response(user_query, kb_entry, lang_mode)
    
    def _generate_ai_response(self, user_query: str, kb_entry: Dict[str, Any], lang_mode: str) -> str:
        """Use AI to generate response when template doesn't fit"""
        facts = kb_entry.get("answer_facts", {})
        kb_text = "\n".join(f"{key}: {value}" for key, value in facts.items())
        
        if lang_mode == "en":
            system_prompt = """You are Sarvajna, a friendly voice assistant for LBS College.
            
RULES:
1. Answer ONLY what was asked - don't give extra information
2. Keep response under 25 words
3. Sound natural and friendly
4. No bullet points or lists
5. Perfect for text-to-speech"""
        else:
            system_prompt = """You are Sarvajna, a friendly Malayalam voice assistant.
            
RULES:
1. Answer ONLY the specific question
2. Keep response short (under 25 words)
3. Use simple Malayalam or Manglish
4. Sound natural and friendly"""
        
        user_message = f"""Question: {user_query}

Data:
{kb_text}

Give a SHORT, DIRECT answer to ONLY what was asked."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.4,
                max_tokens=80
            )
            return self._clean_for_tts(response.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"AI Error: {e}")
            return list(facts.values())[0] if facts else "Sorry, I couldn't find that information."
    
    def _clean_for_tts(self, text: str) -> str:
        """Clean text for TTS output"""
        text = text.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
        text = re.sub(r'^[-•●]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+[.)]\s*', '', text, flags=re.MULTILINE)
        text = ' '.join(text.split())
        return text.strip()
    
    def generate_not_found_response(self, lang_mode: str) -> str:
        if lang_mode in ["ml_script", "manglish"]:
            responses = [
                "ക്ഷമിക്കണം, ആ വിവരം എന്റെ കയ്യിൽ ഇല്ല. മറ്റെന്തെങ്കിലും ചോദിക്കൂ!",
                "അത് എനിക്ക് അറിയില്ല. വേറെ എന്തെങ്കിലും ചോദിക്കാമോ?",
            ]
        else:
            responses = [
                "Sorry, I don't have that specific information. Try asking something else!",
                "I couldn't find that. Can I help with something else about the college?",
            ]
        return random.choice(responses)


# -------------------------------
# SARVAM AI TTS PROCESSOR
# -------------------------------
class AudioProcessor:
    def __init__(self):
        self.client = sarvam_client
        self.language_codes = {
            "en": "en-IN",
            "ml": "ml-IN",
        }
        self.audio_cache = {}
    
    def get_pace_value(self, speech_rate: str) -> float:
        return {"Slow": 0.85, "Normal": 1.0, "Fast": 1.15}.get(speech_rate, 1.0)
    
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
        try:
            if not text or not text.strip():
                return None
            
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
        replacements = {"@": " at ", "&": " and ", "%": " percent ", "+": " plus ", "₹": " rupees "}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return ' '.join(text.split())
    
    def _extract_audio(self, response) -> Optional[bytes]:
        try:
            if hasattr(response, 'audios') and response.audios:
                audio_data = response.audios[0]
                return base64.b64decode(audio_data) if isinstance(audio_data, str) else audio_data
            elif hasattr(response, 'audio'):
                return base64.b64decode(response.audio) if isinstance(response.audio, str) else response.audio
            elif isinstance(response, (str, bytes)):
                return base64.b64decode(response) if isinstance(response, str) else response
        except Exception as e:
            st.error(f"Audio extraction error: {e}")
        return None
    
    def create_autoplay_html(self, audio_bytes: bytes) -> str:
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return f'''<audio id="voiceResponse" autoplay controls style="width:100%;">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>'''


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
        "kb": None, "lh": None, "ai": None, "ap": None, "ch": None,
        "last_audio": None, "autoplay_pending": False, "welcomed": False,
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
    if st.session_state.ch is None:
        st.session_state.ch = ConversationHandler()


# -------------------------------
# PAGE CONFIG & STYLES
# -------------------------------
st.set_page_config(page_title="സർവജ്ഞ – Voice Assistant", page_icon="🎤", layout="wide")

st.markdown("""
<style>
.header { background: linear-gradient(135deg, #1a5276, #2e86ab); color: white; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px; }
.user-msg { background: linear-gradient(135deg, #e8f5e9, #f1f8e9); padding: 15px; border-radius: 12px; border-left: 4px solid #4caf50; margin: 10px 0; }
.bot-msg { background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 15px; border-radius: 12px; border-left: 4px solid #2196f3; margin: 10px 0; }
.voice-box { background: linear-gradient(135deg, #fff3e0, #ffe0b2); padding: 25px; border-radius: 15px; border: 2px dashed #ff9800; text-align: center; margin: 15px 0; }
.welcome-box { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); padding: 20px; border-radius: 15px; border-left: 5px solid #4caf50; margin: 15px 0; }
</style>
""", unsafe_allow_html=True)

init_session()


# -------------------------------
# CORE FUNCTIONS
# -------------------------------
def process_query(query: str, is_voice: bool = False):
    """Process user query and generate voice response"""
    
    # Check if it's conversational
    is_conv, conv_response, conv_type = st.session_state.ch.is_conversation_query(query)
    
    if is_conv:
        response = conv_response
    else:
        lang_mode = st.session_state.lh.detect_language_mode(query)
        processed_query = st.session_state.lh.malayalam_to_manglish(query) if lang_mode == "ml_script" else query
        
        # Search knowledge base
        kb_entry = st.session_state.kb.get_relevant_info(processed_query)
        
        if kb_entry:
            # Extract specific answer for the question
            specific_answer = st.session_state.kb.extract_specific_answer(processed_query, kb_entry)
            response = st.session_state.ai.generate_voice_response(
                processed_query, kb_entry, lang_mode, specific_answer
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
            pace=st.session_state.ap.get_pace_value(prefs["speech_rate"]),
            loudness=prefs["loudness"]
        )
    
    # Save messages
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": query, "time": timestamp, "is_voice": is_voice})
    st.session_state.messages.append({"role": "assistant", "content": response, "time": timestamp, "audio": audio_bytes})
    
    if audio_bytes and st.session_state.preferences["auto_play"]:
        st.session_state.last_audio = audio_bytes
        st.session_state.autoplay_pending = True


def generate_welcome():
    lang = st.session_state.preferences["tts_language"]
    welcome_text = st.session_state.ch.get_welcome_message(lang)
    timestamp = datetime.now().strftime("%H:%M")
    
    audio_bytes = None
    if st.session_state.preferences["voice_enabled"]:
        prefs = st.session_state.preferences
        audio_bytes = st.session_state.ap.text_to_speech(
            welcome_text, prefs["tts_language"], prefs["speaker"],
            pace=st.session_state.ap.get_pace_value(prefs["speech_rate"])
        )
    
    st.session_state.messages.append({"role": "assistant", "content": welcome_text, "time": timestamp, "audio": audio_bytes, "is_welcome": True})
    
    if audio_bytes and st.session_state.preferences["auto_play"]:
        st.session_state.last_audio = audio_bytes
        st.session_state.autoplay_pending = True
    
    st.session_state.welcomed = True


def clear_chat():
    st.session_state.messages = []
    st.session_state.last_audio = None
    st.session_state.autoplay_pending = False
    st.session_state.welcomed = False


# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.markdown("## 🎤 സർവജ്ഞ")
    st.caption("LBS College Voice Assistant")
    st.divider()
    
    st.markdown("### 🔊 Voice Settings")
    st.session_state.preferences["auto_play"] = st.checkbox("Auto-play responses", value=st.session_state.preferences["auto_play"])
    st.session_state.preferences["voice_enabled"] = st.checkbox("Enable voice", value=st.session_state.preferences["voice_enabled"])
    
    if st.session_state.preferences["voice_enabled"]:
        lang = st.radio("Language", ["Malayalam", "English"], index=0 if st.session_state.preferences["tts_language"] == "ml" else 1)
        st.session_state.preferences["tts_language"] = "ml" if lang == "Malayalam" else "en"
        
        st.session_state.preferences["speaker"] = st.selectbox("Voice", ["arya", "meera", "pavithra", "maitreyi"])
        st.session_state.preferences["speech_rate"] = st.select_slider("Speed", ["Slow", "Normal", "Fast"], value=st.session_state.preferences["speech_rate"])
    
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        clear_chat()
        st.rerun()
    if st.button("👋 Say Welcome", use_container_width=True):
        generate_welcome()
        st.rerun()
    
    st.divider()
    st.markdown("### 💬 Try asking:")
    st.markdown("- *Phone number?*\n- *Where is college?*\n- *Fee details?*\n- *Sugamano?*")


# -------------------------------
# MAIN CONTENT
# -------------------------------
st.markdown('''<div class="header"><h1>🎤 സർവജ്ഞ</h1><p>Your Friendly LBS College Voice Assistant</p></div>''', unsafe_allow_html=True)

# Auto-welcome
if not st.session_state.welcomed and not st.session_state.messages:
    generate_welcome()
    st.rerun()

# Status bar
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Voice:** {'🟢 On' if st.session_state.preferences['voice_enabled'] else '🔴 Off'}")
with col2:
    st.markdown(f"**Language:** {'മലയാളം' if st.session_state.preferences['tts_language'] == 'ml' else 'English'}")
with col3:
    st.markdown(f"**Voice:** {st.session_state.preferences['speaker'].title()}")

st.divider()

# Main layout
main_col, side_col = st.columns([3, 1])

with main_col:
    tab1, tab2 = st.tabs(["💬 Type", "🎤 Speak"])
    
    with tab1:
        user_input = st.text_input("Chat with me:", placeholder="Ask anything! e.g., 'What is the phone number?' or 'Sugamano?'")
        col_send, col_greet = st.columns([3, 1])
        with col_send:
            if st.button("📤 Send", type="primary", use_container_width=True) and user_input.strip():
                process_query(user_input.strip())
                st.rerun()
        with col_greet:
            if st.button("👋 Hi!", use_container_width=True):
                process_query("Hello!")
                st.rerun()
    
    with tab2:
        st.markdown('<div class="voice-box"><h4>🎙️ Speak in Malayalam!</h4><p>Click to record your question</p></div>', unsafe_allow_html=True)
        voice_text = speech_to_text(language='ml-IN', start_prompt="🎙️ Record", stop_prompt="⏹️ Stop", just_once=True, use_container_width=True, key="voice_input")
        if voice_text:
            st.info(f"🎤 I heard: {voice_text}")
            process_query(voice_text, is_voice=True)
            st.rerun()
    
    st.divider()
    st.markdown("### 💬 Conversation")
    
    if not st.session_state.messages:
        st.markdown('<div class="welcome-box"><h4>👋 Hello!</h4><p>Ask me about LBS College - phone, email, courses, fees, anything!</p></div>', unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            icon = "🎤" if msg.get("is_voice") else "💬"
            st.markdown(f'<div class="user-msg"><strong>You {icon}</strong> ({msg["time"]})<br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg"><strong>🤖 സർവജ്ഞ</strong> ({msg["time"]})<br>{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/wav")
    
    if st.session_state.autoplay_pending and st.session_state.last_audio:
        st.session_state.autoplay_pending = False
        st.markdown(st.session_state.ap.create_autoplay_html(st.session_state.last_audio), unsafe_allow_html=True)

with side_col:
    st.markdown("### 💬 Quick Chat")
    for label, query in [("👋 Hello!", "Hello!"), ("🙏 നമസ്കാരം", "നമസ്കാരം"), ("😊 Sugamano?", "Sugamano?")]:
        if st.button(label, key=f"greet_{label}", use_container_width=True):
            process_query(query)
            st.rerun()
    
    st.divider()
    st.markdown("### ⚡ Quick Questions")
    
    questions = [
        ("📞 Phone?", "What is the phone number?"),
        ("📍 Location?", "Where is the college located?"),
        ("📧 Email?", "What is the email address?"),
        ("💰 Fees?", "What are the fees?"),
        ("📚 Courses?", "What courses are available?"),
        ("🏠 Hostel?", "Tell me about hostel"),
        ("💼 Placement?", "What about placements?"),
        ("🕐 Timing?", "What is the college timing?"),
    ]
    
    for label, question in questions:
        if st.button(label, key=f"q_{label}", use_container_width=True):
            process_query(question)
            st.rerun()

# Footer
st.divider()
st.markdown('<div style="text-align:center; color:#666; padding:15px;"><p>🎓 <strong>LBS College of Engineering, Kasaragod</strong></p><p>🔊 Powered by <strong>Sarvam AI</strong></p></div>', unsafe_allow_html=True)
