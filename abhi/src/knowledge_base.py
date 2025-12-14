import os
import json
import difflib
from typing import Optional, Dict, Any


class KnowledgeBase:
    def __init__(self, file_name: str = "faq_data.json"):
        self.file_path = file_name
        self.faqs = []
        self.load_faqs()
    
    def load_faqs(self):
        """Load FAQ data from JSON file"""
        try:
            # First check in current directory
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.faqs = json.load(f)
            # Check in data directory
            elif os.path.exists(f"data/{self.file_path}"):
                with open(f"data/{self.file_path}", "r", encoding="utf-8") as f:
                    self.faqs = json.load(f)
            else:
                print(f"Warning: Knowledge base file '{self.file_path}' not found.")
                self.faqs = []
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.faqs = []
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison"""
        return text.lower().strip()
    
    def get_relevant_info(self, query: str) -> Optional[Dict[str, Any]]:
        """Get relevant information from knowledge base"""
        if not self.faqs:
            return None
        
        q_norm = self._normalize(query)
        
        # 1) Exact / substring on question_patterns
        for entry in self.faqs:
            for pattern in entry.get("question_patterns", []):
                p_norm = self._normalize(pattern)
                if p_norm in q_norm or q_norm in p_norm:
                    return entry
        
        # 2) Fuzzy matching on tags
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