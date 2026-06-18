import time
import argparse
import logging
import scipy.io.wavfile
import sounddevice as sd
import config

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Recorder")

def record_audio(output_path: str, duration: int = 5, sample_rate: int = config.SAMPLE_RATE):
    """
    Enregistre de l'audio depuis le microphone et le sauvegarde au format WAV à 16kHz.
    
    Args:
        output_path (str): Chemin vers le fichier WAV de sortie.
        duration (int): Durée de l'enregistrement en secondes.
        sample_rate (int): Taux d'échantillonnage de l'enregistrement.
    """
    try:
        logger.info("=== Liste des périphériques audio disponibles ===")
        logger.info(f"\n{sd.query_devices()}")
        
        logger.info(f"Préparation de l'enregistrement : {duration} secondes à {sample_rate} Hz...")
        
        # Enregistrement audio (mono canal, 1 canal de capture pour la voix)
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        
        logger.info(">>> Enregistrement EN COURS... Parlez dans votre microphone.")
        
        # Attente de la fin de l'enregistrement
        sd.wait()
        
        logger.info("<<< Enregistrement TERMINÉ.")
        
        # Sauvegarde au format WAV
        logger.info(f"Sauvegarde dans {output_path}...")
        scipy.io.wavfile.write(output_path, sample_rate, recording)
        logger.info("Fichier audio sauvegardé avec succès.")
        
    except Exception as e:
        logger.error(f"Erreur d'enregistrement micro : {str(e)}")
        logger.error("Assurez-vous qu'un périphérique de capture est connecté et que la bibliothèque portaudio est installée.")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enregistre de l'audio à partir du microphone.")
    parser.add_argument("-o", "--output", type=str, default="input_record.wav", help="Chemin du fichier audio de sortie")
    parser.add_argument("-d", "--duration", type=int, default=5, help="Durée de l'enregistrement en secondes")
    args = parser.parse_args()
    
    record_audio(args.output, args.duration)
