import torch

# Détection automatique du CPU/GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration des modèles Hugging Face
# whisper-tiny est idéal pour un CPU local car il est léger (~150 Mo) et rapide.
# Pour de meilleures performances, utiliser 'openai/whisper-base' ou 'openai/whisper-large-v3'.
STT_MODEL = "openai/whisper-tiny"

# google/flan-t5-base (~250M paramètres) est optimal pour les instructions textuelles sur CPU local.
# Alternatives : 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' (pour un LLM de chat) ou 'Helsinki-NLP/opus-mt-fr-en' (traduction).
T2T_MODEL = "google/flan-t5-base"

# facebook/mms-tts-fra est un modèle VITS très léger et rapide pour la génération de voix en français.
# facebook/mms-tts-eng est disponible pour l'anglais.
TTS_MODEL = "facebook/mms-tts-fra"

# Paramètres audio
SAMPLE_RATE = 16000  # Fréquence d'échantillonnage standard exigée par Whisper et MMS-TTS
