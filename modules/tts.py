import logging
import torch
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer
import config

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTS")

class TextToSpeech:
    """
    Classe gérant la synthèse vocale du texte en audio (Text-to-Speech).
    Utilise le modèle VITS MMS-TTS de Facebook pour un rendu vocal rapide et local.
    """
    def __init__(self, model_name: str = config.TTS_MODEL, device: str = config.DEVICE):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """Charge le modèle et le tokenizer uniquement lors de la première utilisation."""
        if self.model is None or self.tokenizer is None:
            logger.info(f"Chargement du modèle TTS : {self.model_name} sur {self.device}...")
            
            # Utilisation des classes spécifiques à VITS pour une meilleure robustesse
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = VitsModel.from_pretrained(self.model_name).to(self.device)
            
            logger.info("Modèle TTS chargé avec succès.")

    def synthesize(self, text: str, output_path: str) -> str:
        """
        Génère un fichier audio WAV à partir d'un texte d'entrée.
        
        Args:
            text (str): Le texte à convertir en voix.
            output_path (str): Le chemin de destination du fichier audio généré (WAV).
            
        Returns:
            str: Le chemin vers le fichier audio créé.
        """
        if not text:
            raise ValueError("Le texte à synthétiser est vide.")
            
        self._load_model()
        try:
            logger.info(f"Synthèse vocale pour le texte : '{text}'...")
            
            # Préparation des inputs pour le modèle
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Génération de l'audio sans calcul de gradient pour accélérer et limiter la mémoire
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Récupération de la waveform générée par le modèle (sur CPU en numpy)
            # outputs.waveform contient l'audio généré (généralement à 16000Hz)
            # MMS-TTS renvoie parfois .waveform ou .audio selon la version de transformers
            waveform = None
            if hasattr(outputs, "waveform"):
                waveform = outputs.waveform[0].cpu().numpy()
            elif hasattr(outputs, "audio"):
                waveform = outputs.audio[0].cpu().numpy()
            else:
                # Fallback générique si les attributs diffèrent
                waveform = outputs[0][0].cpu().numpy()
            
            # Récupération du taux d'échantillonnage configuré dans le modèle (ou par défaut 16000Hz)
            sampling_rate = getattr(self.model.config, "sampling_rate", config.SAMPLE_RATE)
            
            # Écriture du fichier audio au format WAV
            logger.info(f"Sauvegarde du fichier audio généré dans : {output_path} (Taux : {sampling_rate} Hz)...")
            scipy.io.wavfile.write(output_path, rate=sampling_rate, data=waveform)
            
            logger.info("Synthèse vocale terminée.")
            return output_path
        except Exception as e:
            logger.error(f"Erreur lors de la synthèse vocale : {str(e)}")
            raise e
