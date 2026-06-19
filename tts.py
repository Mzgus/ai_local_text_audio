import scipy.io.wavfile
from transformers import pipeline

TTS_MODEL = "facebook/mms-tts-eng"

def synthesize(texte: str, output_path: str, device: str = "cpu") -> None:
    """
    Convertit un texte en fichier audio WAV.
    """
    if not texte:
        raise ValueError("Le texte à synthétiser est vide.")

    # Nettoyage rapide du texte
    for orig, rep in [("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")]:
        texte = texte.replace(orig, rep)
    texte = " ".join(texte.split())

    print(f"[TTS] Chargement du modèle {TTS_MODEL} sur {device}...")
    tts_pipeline = pipeline("text-to-speech", model=TTS_MODEL, device=device)

    print(f"[TTS] Génération vocale pour : {texte}")
    output = tts_pipeline(texte)

    print(f"[TTS] Sauvegarde dans : {output_path}")
    scipy.io.wavfile.write(output_path, rate=output["sampling_rate"], data=output["audio"].T)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python tts.py \"<texte>\" [fichier_sortie.wav] [device]")
        sys.exit(1)

    texte = sys.argv[1]
    fichier_sortie = sys.argv[2] if len(sys.argv) > 2 else "output.wav"
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

    synthesize(texte, fichier_sortie, device=device)
