"""
AI processing module using Perplexity API
"""

import os
from openai import OpenAI
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AIProcessor:
    """Handles AI interactions using Perplexity API"""
    
    def __init__(self, model: str = "sonar"):
        self.model = model
        self.api_key = os.getenv("PPLX_API_KEY")
        
        if not self.api_key:
            raise ValueError("PPLX_API_KEY not found in environment variables")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai",
        )
    
    def rewrite_from_kb(self, user_query: str, kb_entry: Dict[str, Any], lang_mode: str) -> str:
        """
        Generate natural response from knowledge base facts
        
        Args:
            user_query: User's question
            kb_entry: Knowledge base entry
            lang_mode: Language mode (en, manglish)
            
        Returns:
            Natural language response
        """
        facts = kb_entry.get("answer_facts", {})
        tags = kb_entry.get("tags", [])
        
        # Format facts as text
        kb_text = "\n".join(f"{key}: {value}" for key, value in facts.items())
        
        # Language-specific instructions
        if lang_mode == "en":
            lang_instruction = """
            Answer in simple, clear English. 
            Use a friendly, helpful tone suitable for college students.
            """
        else:
            lang_instruction = """
            Answer in Manglish (Malayalam written in English letters).
            Use a warm, natural, conversational tone.
            Include common Malayalam expressions where appropriate.
            """
        
        # System prompt
        system_prompt = f"""
        You are സർവജ്ഞ (Sarva-jña), an AI assistant for LBS College of Engineering, Kasaragod.
        
        CRITICAL RULES:
        1. Use ONLY the information provided in KB_FACTS
        2. Do NOT invent new facts or details
        3. Do NOT use internet knowledge
        4. Structure your response naturally, not as a list
        5. {lang_instruction}
        
        If KB_FACTS don't fully answer the question, acknowledge this politely.
        """
        
        # User message
        user_message = f"""
        User Question: {user_query}
        
        Relevant Tags: {', '.join(tags)}
        
        KB Facts:
        {kb_text}
        
        Please provide a helpful answer based only on the KB facts above.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"AI Processing Error: {e}")
            
            # Fallback response
            if lang_mode == "en":
                return f"Here's what I know: {' '.join(facts.values())}"
            else:
                return f"ഇതാ എനിക്കറിയാവുന്നത്: {' '.join(facts.values())}"
    
    def generate_general_response(self, query: str, lang_mode: str) -> str:
        """
        Generate general response when KB doesn't have answer
        
        Args:
            query: User query
            lang_mode: Language mode
            
        Returns:
            General response
        """
        if lang_mode == "en":
            return """
            I'm sorry, I don't have specific information about that in my knowledge base. 
            For detailed information about this topic, please:
            1. Visit the college website: lbscek.ac.in
            2. Contact the administration office
            3. Visit the college in person
            
            Is there anything else I can help you with?
            """
        else:
            return """
            ക്ഷമിക്കണം, ഈ വിഷയത്തെക്കുറിച്ചുള്ള വിശദമായ വിവരങ്ങൾ എന്റെ ഡാറ്റാബേസിൽ ഇല്ല.
            ഇതിനെക്കുറിച്ചുള്ള വിവരങ്ങൾക്കായി ദയവായി:
            1. കോളേജ് വെബ്സൈറ്റ് സന്ദർശിക്കുക: lbscek.ac.in
            2. ഓഫീസുമായി ബന്ധപ്പെടുക
            3. കോളേജിൽ സ്വയം വരിക
            
            മറ്റെന്തെങ്കിലും സഹായിക്കാനാണോ?
            """