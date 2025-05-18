from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import uuid

# === Import modul STT, LLM, TTS ===
from stt import transcribe_speech_to_text
from llm import generate_response
from tts import transcribe_text_to_speech

# === Import dan inisialisasi G2P (Grapheme to Phoneme) ===
from g2p_id import G2P
g2p = G2P()  # Inisialisasi hanya sekali agar lebih efisien

# === Inisialisasi FastAPI ===
app = FastAPI()

@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    # === 1. Baca dan simpan file audio sementara ===
    file_bytes = await audio.read()
    file_ext = os.path.splitext(audio.filename)[-1] or ".wav"

    # === 2. Transkripsi audio ke teks (Speech-to-Text / STT) ===
    user_text = transcribe_speech_to_text(file_bytes, file_ext=file_ext)
    if "[ERROR]" in user_text:
        return {"error": "Gagal transkripsi audio", "detail": user_text}
    print(f"[STT] Transkripsi: {user_text.strip()}")

    # === 3. Kirim teks hasil STT ke LLM untuk mendapatkan respons ===
    response_text = generate_response(user_text.strip())
    if "[ERROR]" in response_text:
        return {"error": "Gagal mendapatkan respons dari LLM", "detail": response_text}
    print(f"[LLM] Balasan: {response_text}")

    # === 4. Ubah teks respons ke fonem menggunakan G2P (opsional, tergantung TTS) ===
    phonemes = g2p(response_text)
    print(f"[G2P] Fonem: {phonemes}")

    # === 5. Konversi fonem ke audio (Text-to-Speech / TTS) ===
    audio_response_path = transcribe_text_to_speech(phonemes)

    if not os.path.isfile(audio_response_path):
        return {"error": "Gagal mengubah teks ke suara"}

    # === 6. Kirim file audio hasil TTS ke klien ===
    return FileResponse(audio_response_path, media_type="audio/wav", filename="response.wav")