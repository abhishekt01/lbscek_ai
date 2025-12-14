# 🎓 LBS College AI Voice Assistant - സർവജ്ഞ

A multilingual voice-enabled AI assistant for LBS College of Engineering, Kasaragod.

## 🌟 Features
- 🎤 Voice input in Malayalam/English
- 🔊 Auto-play voice responses
- 🌐 Multi-language support (Malayalam/English/Manglish)
- 🎓 College-specific knowledge base
- 💬 Text & Voice input options
- 📱 Responsive web interface

## 🚀 Quick Start

### Local Development
```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/lbs-voice-assistant.git
cd lbs-voice-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your Perplexity API key

# 5. Run the app
streamlit run app.py
