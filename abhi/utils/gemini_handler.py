import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class GeminiAIProcessor:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")
        genai.configure(api_key=GEMINI_API_KEY)
        # Use a fast model; you can change to a different one if needed
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_rewritten_answer(self, user_query: str, kb_entry: dict, lang_mode: str) -> str:
        """
        lang_mode: 'en', 'ml_script', 'manglish'
        """
        facts = kb_entry.get("answer_facts", {})
        tags = kb_entry.get("tags", [])

        base_facts_str = ""
        for k, v in facts.items():
            base_facts_str += f"{k}: {v}\n"

        if lang_mode == "en":
            lang_instruction = "Answer in simple, clear English."
        elif lang_mode == "ml_script":
            lang_instruction = "Answer in Malayalam script."
        else:
            lang_instruction = (
                "Answer in Manglish (Malayalam written in English letters), "
                "using friendly, natural tone."
            )

        system_prompt = (
            "You are an AI assistant for LBS College of Engineering, Kasaragod.\n"
            "You MUST use only the information given in KB_FACTS when stating facts.\n"
            "Do not invent new factual details. You can change wording and style.\n"
            "Do not repeat KB_FACTS as a list; create a short, natural answer.\n"
        )

        full_prompt = f"""
{system_prompt}

User Question:
{user_query}

KB_TAGS:
{tags}

KB_FACTS:
{base_facts_str}

Language requirement:
{lang_instruction}
"""

        response = self.model.generate_content(full_prompt)
        return response.text.strip()
