import logging
from transformers import pipeline
import config

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STT")

class SpeechToText:
    """
    Classe gérant la transcription de l'audio en texte (Speech-to-Text).
    Utilise le modèle Whisper via la pipeline Hugging Face.
    """
    def __init__(self, model_name: str = config.STT_MODEL, device: str = config.DEVICE):
        self.model_name = model_name
        self.device = device
        self.pipe = None

    def _load_model(self):
        """Charge le modèle uniquement lors de la première utilisation pour économiser la RAM."""
        if self.pipe is None:
            logger.info(f"Chargement du modèle STT : {self.model_name} sur {self.device}...")
            # automatic-speech-recognition gère automatiquement le décodage et le rééchantillonnage de l'audio
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self.device
            )
            logger.info("Modèle STT chargé avec succès.")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcrit un fichier audio donné en texte.
        
        Args:
            audio_path (str): Chemin vers le fichier audio (WAV, MP3, etc.).
            
        Returns:
            str: Texte transcrit.
        """
        self._load_model()
        try:
            logger.info(f"Transcription du fichier audio : {audio_path}...")
            # Whisper génère un dictionnaire contenant la clé 'text'
            result = self.pipe(audio_path)
            transcription = result.get("text", "").strip()
            logger.info("Transcription terminée.")
            return transcription
        except Exception as e:
            logger.error(f"Erreur lors de la transcription : {str(e)}")
            raise e
