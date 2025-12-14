import json
import os
import difflib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class KnowledgeBase:
    def __init__(self, file_name: str = "faq_data.json"):
        self.file_path = os.path.join(DATA_DIR, file_name)
        self.faqs = []
        self.load_faqs()

    def load_faqs(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.faqs = json.load(f)
        except Exception as e:
            print("Error loading KB:", e)
            self.faqs = []

    def _normalize(self, text: str) -> str:
        return text.lower().strip()

    def get_relevant_info(self, query: str):
        """
        Very simple matcher:
        1) exact / substring match on patterns
        2) fuzzy match on tags
        Returns the best KB entry or None.
        """
        if not self.faqs:
            return None

        q_norm = self._normalize(query)

        # 1. Exact / substring on question_patterns
        best_entry = None
        best_score = 0.0

        for entry in self.faqs:
            for pattern in entry.get("question_patterns", []):
                p_norm = self._normalize(pattern)
                if p_norm in q_norm or q_norm in p_norm:
                    return entry

        # 2. Fuzzy on tags
        all_tags = []
        tag_to_entry = {}
        for entry in self.faqs:
            for t in entry.get("tags", []):
                all_tags.append(t)
                tag_to_entry[t] = entry

        close = difflib.get_close_matches(q_norm, all_tags, n=1, cutoff=0.5)
        if close:
            return tag_to_entry[close[0]]

        return None
