#!/usr/bin/env python3
"""
Minimal web server that uses app.py as the AI core and serves an HTML/CSS UI.
No changes to app.py required.
"""
import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
import io

# Import core from app.py
from app import get_llm_response, transcribe

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(silent=True) or {}
    user_text = (data.get('text') or '').strip()
    if not user_text:
        return jsonify({'error': 'text is required'}), 400
    try:
        reply = get_llm_response(user_text)
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file is required'}), 400
    f = request.files['audio']
    if not f.filename:
        return jsonify({'error': 'empty filename'}), 400
    suffix = os.path.splitext(f.filename)[1] or '.webm'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        path = tmp.name
        f.save(path)
    try:
        # Decode via ffmpeg/pydub with fallback to soundfile
        audio_np = None
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(path)
            if seg.channels > 1:
                seg = seg.set_channels(1)
            seg = seg.set_frame_rate(16000)
            samples = seg.get_array_of_samples()
            audio_np = np.asarray(samples, dtype=np.float32) / (2 ** (8 * seg.sample_width - 1))
        except Exception:
            try:
                import soundfile as sf
                audio_np, sr = sf.read(path, dtype='float32', always_2d=False)
                if audio_np.ndim > 1:
                    audio_np = np.mean(audio_np, axis=1)
            except Exception as dec_err:
                return jsonify({'error': f'Failed to decode audio. Install ffmpeg. Details: {dec_err}'}), 400
        text = transcribe(audio_np) if audio_np is not None else ''
        reply = get_llm_response(text) if text else "I couldn't hear clearly. Please repeat."
        return jsonify({'transcript': text, 'reply': reply})
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

@app.route('/api/tts', methods=['POST'])
def api_tts():
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'text is required'}), 400
    # Always use gTTS (MP3)
    try:
        from gtts import gTTS
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as m:
            out_mp3 = m.name
        gTTS(text).save(out_mp3)
        with open(out_mp3, 'rb') as f:
            data_bytes = f.read()
        return send_file(io.BytesIO(data_bytes), mimetype='audio/mpeg', as_attachment=False, download_name='tts.mp3')
    except Exception as gerr:
        return jsonify({'error': f'gTTS failed: {gerr}'}), 500
    finally:
        try:
            os.unlink(out_mp3)
        except Exception:
            pass

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    print(f'Web UI running on http://{host}:{port}')
    app.run(host=host, port=port, debug=True)
