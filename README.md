# AI Voice Assistant with Gemini

A conversational voice assistant that uses:
- **Speech-to-Text**: OpenAI Whisper for converting speech to text
- **AI Language Model**: Google Gemini AI for generating responses
- **Text-to-Speech**: Google Text-to-Speech (gTTS) for converting responses to speech
- **Audio Input**: Microphone recording for voice input
- **Audio Output**: Speaker output for voice responses

## Features

- Real-time voice conversation with AI
- Text-based fallback mode
- Conversation memory for context
- High-quality speech synthesis
- Cross-platform audio support

## Setup

1. **Activate the virtual environment**:
   ```bash
   source .ai/bin/activate
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```

3. **Get your Gemini API key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key

4. **Configure the API key**:
   - Edit the `.env` file
   - Replace `your_gemini_api_key_here` with your actual API key

## Usage

1. **Activate the virtual environment**:
   ```bash
   source .ai/bin/activate
   ```

2. **Run the voice assistant**:
   ```bash
   python app.py
   ```

3. **Voice Mode** (if audio is available):
   - Press Enter to start recording
   - Speak your message
   - Press Enter to stop recording
   - The AI will respond with voice

4. **Text Mode** (if audio is not available):
   - Type your message and press Enter
   - The AI will respond in text
   - Type 'quit' to exit

## Requirements

- Python 3.8+
- Microphone (for voice input)
- Speakers (for voice output)
- Internet connection (for Gemini AI and gTTS)
- Google Gemini API key

## Troubleshooting

- **Audio issues**: Make sure your microphone and speakers are working
- **API errors**: Check your Gemini API key in the `.env` file
- **Permission errors**: Make sure the script has execute permissions (`chmod +x setup.sh`)

## Dependencies

- `google-generativeai`: Google Gemini AI integration
- `gtts`: Google Text-to-Speech
- `openai-whisper`: Speech-to-Text
- `sounddevice`: Audio recording
- `pygame`: Audio playback
- `nltk`: Natural language processing
- `python-dotenv`: Environment variable management