from transformers import pipeline

STT_MODEL = "openai/whisper-large-v3"

def transcribe(audio_path: str, device: str = "cpu") -> str:
    """
    Transcrit un fichier audio en texte.
    """
    print(f"[STT] Chargement du modèle {STT_MODEL} sur {device}...")
    stt_pipeline = pipeline("automatic-speech-recognition", model=STT_MODEL, device=device)
    
    print(f"[STT] Transcription en cours pour : {audio_path}")
    result = stt_pipeline(audio_path)
    return result["text"].strip()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python stt.py <chemin_audio> [device]")
        sys.exit(1)

    chemin_audio = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

    texte_transcrit = transcribe(chemin_audio, device=device)
    print(f"\n=== Résultat ===\n{texte_transcrit}")
