import logging
from transformers import pipeline
import config

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM")

class TextToText:
    """
    Classe gérant le traitement du texte et la génération de réponses (Text-to-Text).
    Supporte les modèles Seq2Seq (Flan-T5) ou autoregressifs (TinyLlama/Phi-3).
    """
    def __init__(self, model_name: str = config.T2T_MODEL, device: str = config.DEVICE):
        self.model_name = model_name
        self.device = device
        self.pipe = None
        
    def _load_model(self):
        """Charge le modèle uniquement lors de la première utilisation pour économiser la RAM."""
        if self.pipe is None:
            logger.info(f"Chargement du modèle LLM : {self.model_name} sur {self.device}...")
            
            # Détermination de la pipeline adéquate selon le type de modèle
            if "t5" in self.model_name.lower():
                pipeline_task = "text2text-generation"
            else:
                pipeline_task = "text-generation"
                
            self.pipe = pipeline(
                pipeline_task,
                model=self.model_name,
                device=self.device
            )
            logger.info("Modèle LLM chargé avec succès.")

    def generate_response(self, prompt: str, max_length: int = 128) -> str:
        """
        Génère une réponse textuelle à partir d'un prompt d'entrée.
        
        Args:
            prompt (str): Texte d'entrée (transcription).
            max_length (int): Longueur maximale de la génération.
            
        Returns:
            str: Texte généré.
        """
        self._load_model()
        try:
            logger.info(f"Génération de texte pour l'entrée : '{prompt}'...")
            
            # Configuration spécifique pour Flan-T5 ou les LLMs classiques
            if "t5" in self.model_name.lower():
                # Formater le prompt pour T5 pour de meilleurs résultats
                t5_prompt = f"Answer the following question or respond: {prompt}"
                result = self.pipe(t5_prompt, max_length=max_length, do_sample=True, top_k=50, top_p=0.95)
                response = result[0].get("generated_text", "").strip()
            else:
                # Modèles de génération de texte classiques (CausalLM)
                result = self.pipe(prompt, max_length=max_length, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)
                response = result[0].get("generated_text", "")
                # Retirer le prompt d'origine de la réponse si le modèle le renvoie
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
            
            logger.info(f"Réponse générée : '{response}'")
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la génération de texte : {str(e)}")
            raise e
