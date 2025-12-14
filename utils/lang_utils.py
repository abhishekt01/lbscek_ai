from langdetect import detect


def detect_lang_mode(text: str) -> str:
    """
    Rough language mode:
    - 'en'          -> English
    - 'ml_script'   -> Malayalam Unicode
    - 'manglish'    -> everything else assumed Manglish / code-mix
    """
    text = text.strip()
    if not text:
        return "en"

    # Check Malayalam Unicode range
    for ch in text:
        if "\u0d00" <= ch <= "\u0d7f":
            return "ml_script"

    # Simple heuristic with langdetect
    try:
        code = detect(text)
        if code == "en":
            return "en"
    except Exception:
        pass

    return "manglish"
