import os
import argparse
import logging
from modules.stt import SpeechToText
from modules.llm import TextToText
from modules.tts import TextToSpeech
from record import record_audio
import config

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Pipeline")

def run_pipeline(input_audio_path: str, output_audio_path: str, record_first: bool = False, record_duration: int = 5):
    """
    Exécute le pipeline complet :
    Enregistrement optionnel -> STT (Audio vers Texte) -> LLM (Texte vers Texte) -> TTS (Texte vers Audio).
    
    Args:
        input_audio_path (str): Chemin du fichier audio d'entrée à transcrire.
        output_audio_path (str): Chemin du fichier audio de sortie généré.
        record_first (bool): Si True, enregistre l'audio depuis le microphone avant de lancer le traitement.
        record_duration (int): Durée de l'enregistrement en secondes.
    """
    logger.info("=== Initialisation du Pipeline Local ===")
    
    # 1. Enregistrement optionnel depuis le micro
    if record_first:
        logger.info(f"Étape 0 : Enregistrement micro demandé dans '{input_audio_path}'")
        record_audio(input_audio_path, duration=record_duration)
        
    if not os.path.exists(input_audio_path):
        logger.error(f"Fichier audio d'entrée introuvable : '{input_audio_path}'")
        logger.error("Veuillez fournir un fichier audio valide ou utiliser l'option --record pour enregistrer.")
        return

    # 2. Speech-to-Text (Transcription)
    stt_handler = SpeechToText()
    transcription = stt_handler.transcribe(input_audio_path)
    print(f"\n[STT - Transcription obtenue] :\n{transcription}\n")
    
    if not transcription:
        logger.warning("Aucun texte n'a pu être extrait de l'audio. Arrêt du pipeline.")
        return

    # 3. Text-to-Text (Traitement LLM / Génération)
    llm_handler = TextToText()
    response_text = llm_handler.generate_response(transcription)
    print(f"[LLM - Réponse générée] :\n{response_text}\n")
    
    if not response_text:
        logger.warning("Le modèle LLM n'a généré aucun texte. Arrêt du pipeline.")
        return

    # 4. Text-to-Speech (Synthèse Vocale)
    tts_handler = TextToSpeech()
    tts_handler.synthesize(response_text, output_audio_path)
    print(f"[TTS - Audio de sortie sauvegardé dans] : {output_audio_path}\n")
    
    logger.info("=== Exécution du pipeline terminée avec succès ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline local d'analyse et de génération de texte et audio (STT -> LLM -> TTS)."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="input.wav",
        help="Chemin du fichier audio d'entrée (par défaut: input.wav)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.wav",
        help="Chemin du fichier audio généré en sortie (par défaut: output.wav)"
    )
    parser.add_argument(
        "-r", "--record",
        action="store_true",
        help="Activer l'enregistrement par microphone avant le traitement"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=5,
        help="Durée de l'enregistrement microphone en secondes (par défaut: 5)"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        input_audio_path=args.input,
        output_audio_path=args.output,
        record_first=args.record,
        record_duration=args.duration
    )
