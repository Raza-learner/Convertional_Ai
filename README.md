# 🎤 AI Voice Assistant with Enhanced Audio Processing

A sophisticated conversational voice assistant powered by Google Gemini AI, featuring advanced audio processing, real-time voice activity detection, and high-quality text-to-speech synthesis.

## ✨ Features

### 🎯 Core Functionality
- **Real-time voice conversation** with AI using Google Gemini
- **Advanced speech-to-text** using OpenAI Whisper with optimized parameters
- **High-quality text-to-speech** using Coqui TTS with multiple voice options
- **Intelligent voice activity detection** to minimize false triggers
- **Real-time audio level visualization** for microphone monitoring

### 🔧 Audio Processing
- **Audio enhancement** with noise reduction and amplification
- **DC offset removal** and high-pass filtering (80Hz)
- **Soft clipping** to prevent audio distortion
- **Audio normalization** for consistent levels
- **Microphone calibration** with automated sensitivity testing

### 🧠 AI Integration
- **Google Gemini AI** for intelligent responses
- **Conversation memory** for contextual understanding
- **Multiple TTS models** (Tacotron2, FastSpeech2, etc.)
- **Multiple voice options** with different speakers
- **Language support** for various languages

### 📊 Monitoring & Diagnostics
- **Real-time audio level display** with color-coded indicators
- **Voice activity detection metrics** (RMS, spectral centroid, speech ratio)
- **Microphone calibration tool** with feedback and tips
- **Audio processing statistics** and performance metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Microphone and speakers
- Internet connection
- Google Gemini API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd conAi
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n venv python=3.11
   conda activate venv
   ```

3. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

4. **Get your Gemini API key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key

5. **Configure the API key:**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```

### Usage

1. **Activate the environment:**
   ```bash
   conda activate venv
   ```

2. **Run the voice assistant:**
   ```bash
   python app.py
   ```

3. **Choose your mode:**
   - **Option 1**: Test STT functionality
   - **Option 2**: Test TTS functionality  
   - **Option 3**: Start voice conversation
   - **Option 4**: Start text conversation
   - **Option 5**: Calibrate microphone

## 🎛️ Advanced Features

### Voice Activity Detection
The system uses sophisticated algorithms to detect speech:
- **RMS Energy Analysis**: Detects audio volume levels
- **Spectral Centroid**: Analyzes frequency characteristics
- **Speech Frequency Mask**: Focuses on human speech frequencies (300-3400Hz)
- **Speech Content Ratio**: Calculates percentage of speech-like content

### Audio Enhancement Pipeline
1. **DC Offset Removal**: Eliminates constant voltage offset
2. **High-Pass Filtering**: Removes low-frequency noise (80Hz cutoff)
3. **Amplification**: Boosts audio levels with soft clipping
4. **Normalization**: Ensures consistent audio levels
5. **Type Conversion**: Ensures proper data types for Whisper

### Microphone Calibration
- **3-second recording test** to analyze your microphone
- **Automatic sensitivity detection** with feedback
- **Visual level indicators** (🟢 Good, 🟡 Medium, 🟠 Low, 🔴 Very low)
- **Calibration tips** for optimal performance

## 📋 Requirements

### Core Dependencies
- `google-generativeai` - Google Gemini AI integration
- `openai-whisper` - Speech-to-text processing
- `TTS` - Coqui text-to-speech synthesis
- `sounddevice` - Audio recording and playback
- `pygame` - Audio system management
- `numpy` - Numerical computations
- `scipy` - Signal processing
- `torch` - PyTorch for TTS models

### Optional Dependencies
- `pyaudio` - Alternative audio backend
- `nltk` - Natural language processing
- `python-dotenv` - Environment variable management

## 🔧 Configuration

### Environment Variables
Create a `.env` file with:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Audio Settings
The system automatically detects and configures:
- **Audio backend**: sounddevice (primary) or pyaudio (fallback)
- **Sample rate**: 16000 Hz (optimized for Whisper)
- **Channels**: Mono (1 channel)
- **Buffer size**: 1024 frames

### TTS Models
Available models (automatically downloaded):
- `tts_models/en/ljspeech/tacotron2-DDC` (default)
- `tts_models/en/ljspeech/fastspeech2`
- `tts_models/en/vctk/vits`
- And 70+ other models

## 🐛 Troubleshooting

### Audio Issues
- **No microphone detected**: Check audio permissions and device connections
- **Poor audio quality**: Run microphone calibration (Option 5)
- **Audio not playing**: Check speaker connections and volume levels

### API Issues
- **Gemini API errors**: Verify your API key in `.env` file
- **TTS model errors**: Models are downloaded automatically on first use
- **Whisper errors**: Ensure audio is in correct format (16kHz, mono)

### Performance Issues
- **Slow transcription**: Whisper models are CPU-intensive
- **High memory usage**: TTS models require significant RAM
- **Audio latency**: Check system audio buffer settings

## 📊 Performance Metrics

### Audio Processing
- **Enhancement time**: ~10-50ms per 3-second clip
- **VAD processing**: ~5-20ms per frame
- **Whisper transcription**: 2-10 seconds depending on model
- **TTS synthesis**: 1-5 seconds depending on text length

### System Requirements
- **RAM**: 4GB+ recommended (8GB+ for best performance)
- **CPU**: Multi-core processor recommended
- **Storage**: 2GB+ for models and dependencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech-to-text capabilities
- **Google Gemini** for AI language processing
- **Coqui TTS** for text-to-speech synthesis
- **NumPy/SciPy** for audio signal processing
- **PyTorch** for machine learning model support

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Run microphone calibration (Option 5)
3. Test individual components (Options 1-2)
4. Open an issue on GitHub

---

**Made with ❤️ for the AI community**