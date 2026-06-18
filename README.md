# Pipeline Audio -> Texte -> Audio (STT -> LLM -> TTS)

Ce projet implémente un pipeline d'intelligence artificielle complet s'exécutant localement. Il permet de capturer ou de lire une séquence audio, de la transcrire en texte, de générer une réponse textuelle intelligente grâce à un modèle de langage (LLM), puis de synthétiser cette réponse sous forme de voix parlée.

## 1. Architecture et Conception

L'architecture est modulaire et optimisée pour une exécution locale (CPU/GPU) :

```
ai_local_text_audio/
├── requirements.txt         # Dépendances Python nécessaires
├── setup_env.sh             # Script d'installation de l'environnement virtuel (.venv)
├── config.py                # Configuration centrale (modèles, périphériques)
├── modules/
│   ├── __init__.py          # Fichier de packaging
│   ├── stt.py               # Module Speech-to-Text (Transcription avec Whisper)
│   ├── llm.py               # Module Text-to-Text (Génération avec Flan-T5)
│   └── tts.py               # Module Text-to-Speech (Synthèse vocale avec MMS-TTS)
├── record.py                # Script utilitaire d'enregistrement microphone
├── main.py                  # Script principal d'orchestration
└── README.md                # Documentation utilisateur
```

### Optimisations apportées
* **Chargement Différé (Lazy Loading)** : Les modèles ne sont importés et alloués en mémoire (RAM/VRAM) que lorsque l'étape correspondante du pipeline commence.
* **Détection Dynamique du Matériel** : Sélection automatique du processeur graphique (CUDA/GPU) s'il est disponible pour accélérer les calculs, sinon repli sur le CPU.
* **Légèreté des Modèles par Défaut** :
  * **STT** : `openai/whisper-tiny` (150 Mo, ultra rapide).
  * **LLM** : `google/flan-t5-base` (250M paramètres, idéal pour répondre à des questions sur CPU).
  * **TTS** : `facebook/mms-tts-fra` (modèle VITS rapide, produit une voix française claire).

---

## 2. Installation de l'Environnement

### Dépendances Système (Optionnel - Pour l'enregistrement microphone)
Si vous souhaitez utiliser le microphone en entrée, la bibliothèque `sounddevice` nécessite des dépendances système pour capturer le son :
* **Ubuntu/Debian** :
  ```bash
  sudo apt-get update
  sudo apt-get install -y portaudio19-dev libasound2-dev
  ```
* **macOS** :
  ```bash
  brew install portaudio
  ```

### Initialisation de l'environnement virtuel Python
Le projet automatise la configuration via le script `setup_env.sh` :

```bash
chmod +x setup_env.sh
./setup_env.sh
```

Ce script va :
1. Créer un environnement virtuel dans le dossier `.venv`.
2. Mettre à jour `pip`.
3. Installer les dépendances répertoriées dans `requirements.txt` (`torch`, `transformers`, `numpy`, `soundfile`, `scipy`, `sounddevice`, `accelerate`).

---

## 3. Utilisation du Projet

Activez d'abord l'environnement virtuel créé :
```bash
source .venv/bin/activate
```

### Mode 1 : Exécution complète à partir d'un fichier audio existant
Pour transcrire un fichier WAV existant (ex: `mon_entree.wav`), générer une réponse et la synthétiser dans `mon_output.wav` :
```bash
python main.py --input mon_entree.wav --output mon_output.wav
```

### Mode 2 : Enregistrement par microphone en temps réel
Pour enregistrer votre propre voix (durée de 5 secondes par défaut) puis lancer automatiquement le pipeline complet :
```bash
python main.py --record --input voix_micro.wav --output voix_reponse.wav
```
*Note : Vous pouvez ajuster la durée d'enregistrement avec l'option `--duration` (ex: `--duration 10` pour 10 secondes).*

### Utilitaire : Enregistrement seul
Pour enregistrer un fichier audio séparément sans exécuter le pipeline :
```bash
python record.py --output test_micro.wav --duration 5
```

---

## 4. Personnalisation des Modèles

Vous pouvez facilement modifier les modèles utilisés en éditant le fichier de configuration [config.py](file:///home/Ilan/ecole89/prog/b3/ai_local_text_audio/config.py) :

* **Whisper de plus grande taille** : Remplacez `STT_MODEL = "openai/whisper-tiny"` par `"openai/whisper-base"` ou `"openai/whisper-large-v3"` pour une plus grande précision (nécessite un GPU).
* **LLM conversationnel (Chat)** : Remplacez `T2T_MODEL = "google/flan-t5-base"` par `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`.
* **Synthèse Vocale Anglaise** : Remplacez `TTS_MODEL = "facebook/mms-tts-fra"` par `"facebook/mms-tts-eng"`.
