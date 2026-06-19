# stt.py - Etape 1 : Transcription Audio -> Texte (Speech-to-Text)
# -----------------------------------------------------------------
# Modele utilise : openai/whisper-large-v3
# Modele de grande taille (~3 Go), haute precision, supporte +100 langues.
# ATTENTION : Necessite beaucoup de RAM (6 Go+). Sur CPU c'est lent
#             mais ca fonctionne. Avec GPU c'est nettement plus rapide.

import soundfile as sf
import scipy.signal
import numpy as np
from transformers import pipeline

# --- Constantes ---
STT_MODEL = "openai/whisper-large-v3"
TARGET_SAMPLE_RATE = 16000  # Whisper attend de l'audio a 16 000 Hz


def transcribe(audio_path: str, device: str = "cpu") -> str:
    """
    Transcrit un fichier audio en texte.

    Parametre :
        audio_path : chemin vers le fichier audio (WAV ou MP3)
        device : "cpu" ou "cuda"

    Retourne :
        Le texte prononce dans le fichier audio.
    """
    print(f"[STT] Chargement du modele {STT_MODEL}...")
    print("[STT] (Premier lancement : telechargement ~3 Go, patience...)")

    # torch_dtype=float16 reduit la memoire necessaire de moitie sur GPU
    # sur CPU, float32 est utilise automatiquement
    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model=STT_MODEL,
        device=device,
    )

    print(f"[STT] Lecture du fichier audio : {audio_path}")
    audio_data, sample_rate = sf.read(audio_path)

    # Si l'audio est en stereo (2 canaux), on le convertit en mono
    if len(audio_data.shape) > 1:
        print("[STT] Audio stereo detecte -> conversion en mono...")
        audio_data = audio_data.mean(axis=1)

    # Si la frequence n'est pas 16 000 Hz, on reechantillonne
    if sample_rate != TARGET_SAMPLE_RATE:
        print(f"[STT] Reechantillonnage de {sample_rate} Hz -> {TARGET_SAMPLE_RATE} Hz...")
        num_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / sample_rate)
        audio_data = scipy.signal.resample(audio_data, num_samples)
        sample_rate = TARGET_SAMPLE_RATE

    print("[STT] Transcription en cours...")
    result = stt_pipeline({"raw": audio_data.astype(np.float32), "sampling_rate": sample_rate})
    texte = result["text"].strip()

    print(f"[STT] Transcription obtenue : {texte}")
    return texte


# --- Test autonome : lancer ce fichier seul pour tester la transcription ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python stt.py <chemin_audio> [device]")
        print("Exemple : python stt.py input.mp3 cuda")
        sys.exit(1)

    chemin_audio = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

    texte_transcrit = transcribe(chemin_audio, device=device)
    print("\n=== Resultat ===")
    print(texte_transcrit)
