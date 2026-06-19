# main.py - Script principal : lance le pipeline complet
# --------------------------------------------------------
# Ce script orchestre les 3 etapes dans l'ordre :
#   1. stt.py  : Audio  -> Texte    (transcription avec Whisper)
#   2. llm.py  : Texte  -> Texte    (reponse avec Flan-T5)
#   3. tts.py  : Texte  -> Audio    (synthese vocale avec MMS-TTS)
#
# Usage :
#   python main.py --input input.mp3 --output output.wav

import argparse
from stt import transcribe
from llm import generate
from tts import synthesize


def run_pipeline(input_audio: str, output_audio: str, device: str = "cpu"):
    """
    Execute le pipeline complet Audio -> Texte -> Texte -> Audio.

    Parametres :
        input_audio  : chemin vers le fichier audio d'entree (WAV ou MP3)
        output_audio : chemin vers le fichier audio de sortie genere
        device       : appareil a utiliser ("cpu" ou "cuda")
    """
    print("=" * 50)
    print("  PIPELINE : Audio -> Texte -> Reponse -> Audio")
    print("=" * 50)

    # --- Etape 1 : Speech-to-Text ---
    print("\n[1/3] Transcription de l'audio en texte...")
    texte_transcrit = transcribe(input_audio, device=device)
    print(f"\n>>> Transcription : {texte_transcrit}\n")

    # --- Etape 2 : Text-to-Text ---
    print("[2/3] Generation d'une reponse texte...")
    reponse_texte = generate(texte_transcrit, device=device)
    print(f"\n>>> Reponse : {reponse_texte}\n")

    # --- Etape 3 : Text-to-Speech ---
    print("[3/3] Synthese vocale de la reponse...")
    synthesize(reponse_texte, output_audio, device=device)
    print(f"\n>>> Fichier audio genere : {output_audio}")

    print("\n" + "=" * 50)
    print("  Pipeline termine avec succes !")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline IA local : Audio -> Texte -> Reponse -> Audio"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="input.wav",
        help="Fichier audio d'entree (WAV ou MP3). Par defaut : input.wav"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Fichier audio de sortie a generer. Par defaut : output.wav"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Appareil cible pour l'execution : 'cpu' ou 'cuda'. Par defaut : cpu"
    )

    args = parser.parse_args()
    run_pipeline(args.input, args.output, device=args.device)
