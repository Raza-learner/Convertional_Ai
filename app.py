import time
import threading
import numpy as np
import whisper
from queue import Queue
import google.generativeai as genai
from TTS.api import TTS
import pygame
import os
import tempfile
from dotenv import load_dotenv
import nltk
import torch
import warnings
import logging
 

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

# Suppress gRPC/ALTS warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# Suppress specific gRPC logging
logging.getLogger('google.auth').setLevel(logging.ERROR)
logging.getLogger('google.auth.transport').setLevel(logging.ERROR)
logging.getLogger('google.auth.transport.requests').setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# Try to import sounddevice, fallback to alternative if it fails
try:
    import sounddevice as sd
    # Test if audio actually works
    sd.query_devices()
    AUDIO_AVAILABLE = True
    AUDIO_METHOD = "sounddevice"
    print("✓ Audio recording available (sounddevice)")
except (OSError, Exception) as e:
    print(f"Warning: sounddevice not available: {e}")
    # Try PyAudio as fallback
    try:
        import pyaudio
        AUDIO_AVAILABLE = True
        AUDIO_METHOD = "pyaudio"
        print("✓ Audio recording available (pyaudio fallback)")
    except ImportError:
        print("Falling back to text-only mode. You can still use the AI for text conversations.")
        print("To fix audio: Try installing PortAudio system library: sudo apt-get install portaudio19-dev")
        AUDIO_AVAILABLE = False
        AUDIO_METHOD = None
        sd = None
        pyaudio = None

# Setup console (optional, for better output)
try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = None  # Fallback to print

# Load models (use CPU for stability)
device = "cpu"  # Force CPU to avoid GPU memory issues
stt_model = whisper.load_model("base", device=device)  # 'small' for better accuracy, ~500 MB

# Initialize pygame for audio playback
pygame.mixer.init()

# Initialize Coqui TTS engine
try:
    # List available TTS models
    print("Loading Coqui TTS models...")
    tts_temp = TTS()
    tts_models = tts_temp.models
    print(f"Available TTS models: {len(tts_models)}")
    
    # Use a high-quality English model
    # You can change this to other models like:
    # "tts_models/en/ljspeech/tacotron2-DDC" for LJSpeech
    # "tts_models/en/vctk/vits" for multi-speaker
    # "tts_models/multilingual/multi-dataset/xtts_v2" for multilingual
    
    model_name = "tts_models/en/ljspeech/tacotron2-DDC"  # High quality single speaker
    if model_name in tts_models:
        tts_engine = TTS(model_name)
        TTS_AVAILABLE = True
        print(f"✓ Coqui TTS initialized with model: {model_name}")
    else:
        # Fallback to first available model
        tts_engine = TTS(tts_models[0])
        TTS_AVAILABLE = True
        print(f"✓ Coqui TTS initialized with fallback model: {tts_models[0]}")
    
    # Pre-load the model to avoid delay during first use
    print("Pre-loading TTS model...")
    test_audio = tts_engine.tts("Test")
    print("✓ TTS model pre-loaded successfully")
    
except Exception as e:
    print(f"Warning: Coqui TTS not available: {e}")
    TTS_AVAILABLE = False
    tts_engine = None

# Initialize Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables.")
    print("Please set your Gemini API key in a .env file or as an environment variable.")
    GEMINI_AVAILABLE = False
else:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    GEMINI_AVAILABLE = True
    print("✓ Gemini AI initialized successfully")

# Conversation memory
conversation_history = []

def show_audio_level(audio_data):
    """Show real-time audio level visualization"""
    try:
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
        
        # Calculate RMS level
        rms = np.sqrt(np.mean(audio_np**2))
        
        # Create visual bar (20 characters)
        bar_length = 20
        filled_length = int(rms * bar_length * 10)  # Scale up for visibility
        filled_length = min(filled_length, bar_length)
        
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Color coding
        if rms > 0.1:
            color = "🟢"  # Good level
        elif rms > 0.05:
            color = "🟡"  # Medium level
        elif rms > 0.01:
            color = "🟠"  # Low level
        else:
            color = "🔴"  # Very low level
        
        # Print with carriage return to overwrite
        print(f"\r{color} Audio: [{bar}] {rms:.3f}", end="", flush=True)
        
    except Exception:
        pass  # Ignore errors in visualization

def record_audio(stop_event, data_queue, show_levels=False):
    if not AUDIO_AVAILABLE:
        print("Audio recording not available. Please install audio libraries or use text input.")
        return
    
    if AUDIO_METHOD == "sounddevice":
        def callback(indata, frames, time, status):
            if status and console:
                console.print(status)
            
            # Convert to bytes and add to queue
            audio_bytes = bytes(indata)
            data_queue.put(audio_bytes)
            
            # Show audio level if requested
            if show_levels:
                show_audio_level(audio_bytes)
        
        try:
            with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
                while not stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error during audio recording: {e}")
            print("Please check your microphone permissions and audio drivers.")
    
    elif AUDIO_METHOD == "pyaudio":
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
            
            while not stop_event.is_set():
                data = stream.read(1024)
                data_queue.put(data)
                
                # Show audio level if requested
                if show_levels:
                    show_audio_level(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"Error during audio recording: {e}")
            print("Please check your microphone permissions and audio drivers.")

def enhance_audio(audio_np: np.ndarray) -> np.ndarray:
    """Enhance audio for better speech detection"""
    try:
        # Ensure audio is float32 and contiguous
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        
        # Ensure contiguous array for Whisper
        if not audio_np.flags.c_contiguous:
            audio_np = np.ascontiguousarray(audio_np)
        
        # Remove DC offset
        audio_np = audio_np - np.mean(audio_np)
        
        # Apply high-pass filter to remove low-frequency noise
        from scipy import signal
        if len(audio_np) > 100:  # Only if we have enough samples
            # High-pass filter at 80 Hz to remove low-frequency noise
            nyquist = 8000  # Half of 16kHz sample rate
            cutoff = 80 / nyquist
            b, a = signal.butter(2, cutoff, btype='high')
            # Ensure filter coefficients are float32
            b = b.astype(np.float32)
            a = a.astype(np.float32)
            audio_np = signal.filtfilt(b, a, audio_np)
        
        # Amplify audio (but avoid clipping)
        rms = np.sqrt(np.mean(audio_np**2))
        if rms > 0:
            # Amplify to target RMS of 0.1 (adjustable)
            target_rms = 0.1
            amplification = min(target_rms / rms, 10.0)  # Cap at 10x amplification
            audio_np = audio_np * amplification
            
            # Soft clipping to prevent harsh distortion
            audio_np = np.tanh(audio_np * 0.8) * 0.9
        
        # Normalize
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val
        
        # Final check: ensure float32 and contiguous
        audio_np = np.ascontiguousarray(audio_np.astype(np.float32))
        
        return audio_np
        
    except Exception as e:
        print(f"Audio enhancement error: {e}")
        return audio_np

def detect_voice_activity(audio_np: np.ndarray) -> bool:
    """Detect if audio contains voice activity"""
    try:
        # Ensure proper dtype
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_np**2))
        
        # Calculate spectral centroid (brightness of sound)
        fft = np.fft.fft(audio_np)
        freqs = np.fft.fftfreq(len(audio_np), 1/16000)
        power = np.abs(fft)**2
        
        # Focus on speech frequencies (300-3400 Hz)
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        speech_energy = np.sum(power[speech_mask])
        total_energy = np.sum(power[1:len(power)//2])
        speech_ratio = speech_energy / total_energy if total_energy > 0 else 0
        
        # Voice activity criteria (relaxed for better detection)
        has_voice = (rms > 0.001 and  # Lower minimum energy threshold
                    speech_ratio > 0.05 and  # Lower speech content threshold (5% instead of 10%)
                    rms < 0.8)  # Higher max threshold to avoid clipping
        
        # Debug information
        print(f"Voice detection: RMS={rms:.4f}, Speech ratio={speech_ratio:.3f}, Has voice={has_voice}")
        
        return has_voice
        
    except Exception as e:
        print(f"Voice activity detection error: {e}")
        return False

def transcribe(audio_np: np.ndarray) -> str:
    try:
        # Check if audio has sufficient length
        if len(audio_np) < 1600:  # Less than 0.1 seconds at 16kHz
            return ""
        
        # Ensure proper dtype and format for Whisper
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        
        # Ensure contiguous array
        if not audio_np.flags.c_contiguous:
            audio_np = np.ascontiguousarray(audio_np)
        
        # Enhance audio for better speech detection
        audio_enhanced = enhance_audio(audio_np)
        
        # Check for voice activity
        if not detect_voice_activity(audio_enhanced):
            print("No voice activity detected")
            return ""
        
        print("Transcribing audio...")
        
        # Ensure audio is in the right format for Whisper
        # Whisper expects float32 audio in range [-1, 1]
        if audio_enhanced.dtype != np.float32:
            audio_enhanced = audio_enhanced.astype(np.float32)
        
        # Clamp values to valid range
        audio_enhanced = np.clip(audio_enhanced, -1.0, 1.0)
        
        # Use optimized Whisper settings with explicit dtype handling
        result = stt_model.transcribe(
            audio_enhanced, 
            fp16=False,  # Force float32
            language="en",
            temperature=0.0,  # More deterministic
            best_of=1,  # Faster processing
            beam_size=1,  # Faster processing
            patience=1.0,  # Stop early if no progress
            length_penalty=1.0,
            suppress_tokens=[-1],  # Suppress common false tokens
            initial_prompt="Hello, how are you today?"  # Help with context
        )
        
        text = result["text"].strip()
        
        # Enhanced filtering of false positives
        false_positives = [
            "", "you", "thank you", "thanks", "okay", "ok", "yes", "no", 
            "uh", "um", "ah", "eh", "oh", "hmm", "mm", "huh",
            "the", "a", "an", "and", "or", "but", "so", "then"
        ]
        
        if text.lower() in false_positives:
            return ""
        
        # Check if text is too short (likely noise)
        if len(text.split()) < 2:
            return ""
        
        return text
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        # Print more detailed error info for debugging
        print(f"Audio dtype: {audio_np.dtype if 'audio_np' in locals() else 'N/A'}")
        print(f"Audio shape: {audio_np.shape if 'audio_np' in locals() else 'N/A'}")
        return ""

def get_llm_response(text: str) -> str:
    if not GEMINI_AVAILABLE:
        return "Gemini AI is not available. Please check your API key."
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": text})
    
    # Create context from conversation history
    context = ""
    for msg in conversation_history[-10:]:  # Keep last 10 messages for context
        role = "Human" if msg["role"] == "user" else "AI"
        context += f"{role}: {msg['content']}\n"
    
    try:
        # Generate response using Gemini
        response = model.generate_content(f"You are a helpful AI assistant. Keep responses concise and conversational.\n\nConversation history:\n{context}\n\nRespond to the latest message:")
        response_text = response.text.strip()
        
        # Add AI response to conversation history
        conversation_history.append({"role": "assistant", "content": response_text})
        
        return response_text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def synthesize_and_play(text: str, speaker=None, language=None):
    """
    Enhanced Coqui TTS text-to-speech with more options
    
    Args:
        text (str): Text to convert to speech
        speaker (str): Speaker name (for multi-speaker models)
        language (str): Language code (for multilingual models)
    """
    if not TTS_AVAILABLE:
        print(f"AI Response: {text}")
        return
    
    try:
        print(f"Speaking: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Generate audio
        audio = tts_engine.tts(text=text, speaker=speaker, language=language)
        
        # Convert to numpy array if needed
        if isinstance(audio, list):
            audio = np.array(audio)
        
        # Save to temporary file and play
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            # Convert audio to the right format and save
            try:
                import soundfile as sf
                sf.write(tmp_file.name, audio, 22050)  # Coqui TTS typically uses 22050 Hz
            except ImportError:
                # Fallback using scipy if soundfile is not available
                from scipy.io import wavfile
                # Convert to 16-bit PCM
                audio_16bit = (audio * 32767).astype(np.int16)
                wavfile.write(tmp_file.name, 22050, audio_16bit)
            
            # Play the audio file
            pygame.mixer.music.load(tmp_file.name)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
        
        print("✓ Speech completed")
        
    except Exception as e:
        print(f"Error with text-to-speech: {e}")
        print(f"AI Response: {text}")

def synthesize_and_save(text: str, filename: str, speaker=None, language=None):
    """
    Save Coqui TTS audio to a file instead of playing it
    
    Args:
        text (str): Text to convert to speech
        filename (str): Output filename (without extension)
        speaker (str): Speaker name (for multi-speaker models)
        language (str): Language code (for multilingual models)
    """
    if not TTS_AVAILABLE:
        print("TTS not available for file saving")
        return
        
    try:
        print(f"Saving speech to file: {filename}.wav")
        
        # Generate audio
        audio = tts_engine.tts(text=text, speaker=speaker, language=language)
        
        # Convert to numpy array if needed
        if isinstance(audio, list):
            audio = np.array(audio)
        
        # Save to file
        try:
            import soundfile as sf
            sf.write(f"{filename}.wav", audio, 22050)
        except ImportError:
            # Fallback using scipy if soundfile is not available
            from scipy.io import wavfile
            # Convert to 16-bit PCM
            audio_16bit = (audio * 32767).astype(np.int16)
            wavfile.write(f"{filename}.wav", 22050, audio_16bit)
        print(f"✓ Audio saved as {filename}.wav")
        
    except Exception as e:
        print(f"Error saving text-to-speech: {e}")

def get_available_models():
    """Get list of available TTS models"""
    if not TTS_AVAILABLE:
        return {}
    
    try:
        tts_temp = TTS()
        models = tts_temp.models
        model_dict = {}
        
        for i, model in enumerate(models):
            model_dict[i] = {
                'name': model,
                'type': 'single' if 'ljspeech' in model else 'multi' if 'vctk' in model else 'multilingual' if 'multilingual' in model else 'other'
            }
        
        return model_dict
    except Exception as e:
        print(f"Error getting models: {e}")
        return {}

def get_available_speakers():
    """Get list of available speakers (for multi-speaker models)"""
    if not TTS_AVAILABLE:
        return []
    
    try:
        # This depends on the model - some models have speakers, others don't
        # For now, return a generic list
        return ["p225", "p226", "p227", "p228", "p229", "p230"]
    except Exception as e:
        print(f"Error getting speakers: {e}")
        return []

def get_tts_info():
    """Get current TTS model information"""
    if not TTS_AVAILABLE:
        return {}
    
    try:
        return {
            'model_name': str(tts_engine.model_name),
            'available_models': len(TTS.list_models()),
            'available_speakers': len(get_available_speakers())
        }
    except Exception as e:
        print(f"Error getting TTS info: {e}")
        return {}

def test_tts():
    """Test Coqui TTS functionality with different options"""
    print("\n=== Coqui TTS Test Mode ===")
    print("This will test the Text-to-Speech functionality.")
    
    if not TTS_AVAILABLE:
        print("TTS engine not available. Please check your installation.")
        return
    
    test_texts = [
        "Hello! This is a test of the Coqui TTS engine.",
        "I can generate high-quality speech synthesis.",
        "The weather is nice today, isn't it?",
        "Artificial intelligence is fascinating!"
    ]
    
    models = get_available_models()
    print(f"Available models: {len(models)}")
    for i, model in list(models.items())[:5]:  # Show first 5 models
        print(f"  {i}: {model['name']} ({model['type']})")
    
    speakers = get_available_speakers()
    print(f"Available speakers: {len(speakers)}")
    for speaker in speakers[:3]:  # Show first 3 speakers
        print(f"  - {speaker}")
    
    tts_info = get_tts_info()
    print(f"Current TTS info: {tts_info}")
    
    print("\nTesting Coqui TTS with different options:")
    
    # Test 1: Default settings
    print("\n1. Testing default settings:")
    synthesize_and_play(test_texts[0])
    
    # Test 2: Different speakers (if available)
    if speakers:
        print("\n2. Testing different speakers:")
        for speaker in speakers[:2]:  # Test first 2 speakers
            print(f"Testing speaker: {speaker}")
            synthesize_and_play(f"This is speaker {speaker}", speaker=speaker)
    
    # Test 3: Save to file
    print("\n3. Testing save to file:")
    synthesize_and_save(test_texts[2], "test_coqui_speech")
    
    # Test 4: Long text
    print("\n4. Testing long text:")
    long_text = "This is a longer text to test how well the Coqui TTS engine handles extended speech synthesis. It should maintain good quality throughout the entire duration."
    synthesize_and_play(long_text)
    
    print("\n✓ Coqui TTS testing completed!")

def calibrate_microphone():
    """Calibrate microphone for optimal voice detection"""
    print("\n=== Microphone Calibration ===")
    print("This will help optimize your microphone for voice detection.")
    
    if not AUDIO_AVAILABLE:
        print("❌ Audio not available. Cannot calibrate microphone.")
        return False
    
    print("🎤 Testing microphone sensitivity...")
    print("Speak normally for 3 seconds when prompted.")
    
    # Test 1: Normal speaking
    print("\n1. Normal speaking test:")
    input("Press Enter to start...")
    
    data_queue = Queue()
    stop_event = threading.Event()
    recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue, True))
    recording_thread.start()
    
    time.sleep(3)
    stop_event.set()
    recording_thread.join()
    
    print()  # New line after audio level display
    
    audio_data = b"".join(data_queue.queue)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
    
    if audio_np.size > 0:
        rms = np.sqrt(np.mean(audio_np**2))
        has_voice = detect_voice_activity(audio_np)
        
        print(f"📊 Audio level: {rms:.4f}")
        print(f"🎯 Voice activity: {'✅ Detected' if has_voice else '❌ Not detected'}")
        
        if has_voice:
            print("✅ Microphone calibration successful!")
            print("Your microphone is ready for voice conversation.")
            return True
        else:
            print("⚠️ Microphone needs adjustment:")
            print("   • Move closer to microphone (3-6 inches)")
            print("   • Speak louder and more clearly")
            print("   • Reduce background noise")
            return False
    else:
        print("❌ No audio detected. Check microphone connection.")
        return False

def test_stt():
    """Test STT functionality with a simple audio file or manual input"""
    print("\n=== STT Test Mode ===")
    print("This will test the Speech-to-Text functionality.")
    
    if AUDIO_AVAILABLE:
        print("Testing with microphone...")
        input("Press Enter to start recording test...")
        data_queue = Queue()
        stop_event = threading.Event()
        recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue, True))
        recording_thread.start()
        input("Press Enter to stop recording test...")
        stop_event.set()
        recording_thread.join()
        
        audio_data = b"".join(data_queue.queue)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
        
        if audio_np.size > 0:
            print("Transcribing test audio...")
            text = transcribe(audio_np)
            print(f"STT Result: '{text}'")
            if text:
                print("✓ STT is working correctly!")
            else:
                print("⚠ STT returned empty result. Check microphone or speak louder.")
        else:
            print("No audio detected. Check your microphone.")
    else:
        print("Audio not available. Testing with manual text input...")
        test_text = input("Enter test text to process: ").strip()
        if test_text:
            print(f"Processing text: '{test_text}'")
            response = get_llm_response(test_text)
            print(f"AI Response: {response}")
            print("✓ Text processing is working correctly!")

if __name__ == "__main__":
    print("AI Voice Assistant with Gemini started!")
    if not GEMINI_AVAILABLE:
        print("Error: Gemini AI is not available. Please check your API key.")
        exit(1)
    # Ask user what they want to do
    print("\nChoose an option:")
    print("1. Test STT functionality")
    print("2. Test TTS functionality")
    print("3. Start voice conversation")
    print("4. Start text conversation")
    print("5. Calibrate microphone")
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        test_stt()
    elif choice == "2":
        test_tts()
    elif choice == "5":
        calibrate_microphone()
    elif choice == "3" and AUDIO_AVAILABLE:
        print("Mode: Enhanced Voice input/output with Coqui TTS")
        print("🎯 TIPS: Speak clearly, close to microphone, watch audio levels!")
        print("Press Enter to record, Enter again to stop. Ctrl+C to exit.")
        print("=" * 60)
            
        try:
            while True:
                print("\n🎤 Ready to record...")
                print("💡 Speak clearly and watch the audio level indicator!")
                input("Press Enter to start recording...")
                
                data_queue = Queue()
                stop_event = threading.Event()
                recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue, True))
                recording_thread.start()
                
                print("\n🔴 Recording... Speak now! (Watch audio levels)")
                print("🟢 Good level | 🟡 Medium | 🟠 Low | 🔴 Very low")
                input("Press Enter to stop recording...")
                stop_event.set()
                recording_thread.join()
                
                print()  # New line after audio level display
                
                audio_data = b"".join(data_queue.queue)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
                
                if audio_np.size > 0:
                    print("🔄 Processing audio...")
                    
                    # Show audio analysis
                    rms = np.sqrt(np.mean(audio_np**2))
                    print(f"📊 Audio level: {rms:.4f}")
                    
                    # Check for voice activity before transcription
                    if detect_voice_activity(audio_np):
                        print("✅ Voice activity detected, transcribing...")
                        text = transcribe(audio_np)
                        
                        if text:
                            print(f"👤 You: {text}")
                            print("🤖 Generating AI response...")
                            response = get_llm_response(text)
                            print(f"🤖 AI: {response}")
                            
                            if TTS_AVAILABLE:
                                print("🔊 Speaking response with Coqui TTS...")
                                synthesize_and_play(response)
                            else:
                                print("⚠️ TTS not available, showing text only")
                        else:
                            print("❌ Speech detected but couldn't transcribe. Try speaking more clearly.")
                    else:
                        print("❌ No voice activity detected. Try:")
                        print("   • Speaking louder and closer to microphone")
                        print("   • Speaking more clearly")
                        print("   • Reducing background noise")
                else:
                    print("❌ No audio detected. Check your microphone connection.")
        except KeyboardInterrupt:
            print("\n👋 Exiting voice conversation...")
    elif choice == "4" or (choice == "3" and not AUDIO_AVAILABLE):
        print("Mode: Text input/output with Coqui TTS")
        print("Type your message and press Enter. Type 'quit' to exit.")
        print("=" * 50)
        
        try:
            while True:
                text = input("\n👤 You: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if not text:
                    continue
                
                print("🤖 Generating AI response...")
                response = get_llm_response(text)
                print(f"🤖 AI: {response}")
                
                if TTS_AVAILABLE:
                    print("🔊 Speaking response with Coqui TTS...")
                    synthesize_and_play(response)
                else:
                    print("⚠️ TTS not available, showing text only")
        except KeyboardInterrupt:
            print("\n👋 Exiting text conversation...")
    else:
        print("Invalid choice or audio not available for voice mode.")