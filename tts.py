# tts.py - Etape 3 : Synthese Vocale Texte -> Audio (Text-to-Speech)
# -------------------------------------------------------------------
# Modele utilise : facebook/mms-tts-eng
#
# Ce script convertit un texte en fichier audio WAV.
# Modele de taille legere (~150 Mo) s'executant rapidement sur CPU ou GPU.

import scipy.io.wavfile
from transformers import pipeline

# --- Constantes ---
TTS_MODEL = "facebook/mms-tts-eng"


def synthesize(texte: str, output_path: str, device: str = "cpu") -> None:
    """
    Convertit un texte en fichier audio WAV avec le modele MMS-TTS de Meta.

    Parametres :
        texte       : le texte a prononcer (en anglais de preference)
        output_path : chemin du fichier audio de sortie (ex: "output.wav")
        device      : appareil cible ("cpu" ou "cuda")
    """
    if not texte:
        raise ValueError("Le texte a synthetiser est vide.")

    # Nettoyage du texte pour eviter les caracteres speciaux
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
    for orig, rep in replacements.items():
        texte = texte.replace(orig, rep)
    texte = " ".join(texte.split())

    print(f"[TTS] Chargement du modele MMS-TTS ({TTS_MODEL}) sur {device}...")
    
    # Initialisation du pipeline de synthese vocale
    tts_pipeline = pipeline("text-to-speech", model=TTS_MODEL, device=device)

    print(f"[TTS] Generation de la synthese vocale pour : {texte}")
    output = tts_pipeline(texte)

    waveform = output["audio"]
    sampling_rate = output["sampling_rate"]

    # Sauvegarde au format WAV
    print(f"[TTS] Sauvegarde de l'audio dans : {output_path}")
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=waveform.T)
    print(f"[TTS] Fichier audio genere avec succes ({sampling_rate} Hz).")


# --- Test autonome : lancer ce fichier seul pour tester la synthese ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python tts.py \"<texte>\" [fichier_sortie.wav] [device]")
        print("Exemple : python tts.py \"Hello world\" output.wav cpu")
        sys.exit(1)

    texte = sys.argv[1]
    fichier_sortie = sys.argv[2] if len(sys.argv) > 2 else "output.wav"
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

    synthesize(texte, fichier_sortie, device=device)
    print(f"\n=== Resultat : fichier '{fichier_sortie}' cree sur '{device}' ===")
