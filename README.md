# Projet IA - Pipeline Audio -> Texte -> Audio

Ce projet implemente un pipeline d'intelligence artificielle complet qui :
1. **Transcrit** un fichier audio (voix) en texte avec **Whisper Large v3**
2. **Genere** une reponse textuelle avec **Flan-T5**
3. **Synthetise** cette reponse en voix avec **Orpheus 3B**

---

## Structure du projet

```
ai_local_text_audio/
|
|-- stt.py            <- Etape 1 : Audio -> Texte (whisper-large-v3)
|-- llm.py            <- Etape 2 : Texte -> Texte (flan-t5-base)
|-- tts.py            <- Etape 3 : Texte -> Audio (mms-tts-eng)
|
|-- main.py           <- Lance les 3 etapes a la suite
|-- requirements.txt  <- Dependances Python a installer
```

Chaque fichier peut etre lance **seul** pour tester une etape independamment.

---

## Modeles utilises

| Etape | Modele HuggingFace | Taille | Description |
|-------|-------------------|--------|-------------|
| STT | `openai/whisper-large-v3` | ~3 Go | Transcription voix -> texte, haute precision, 100+ langues |
| LLM | `google/flan-t5-base` | ~250 Mo | Generation de reponse textuelle |
| TTS | `facebook/mms-tts-eng` | ~150 Mo | Synthese vocale (anglais) |

> **Note importante** : Le total a telecharger est environ **3.4 Go** lors du premier lancement.
> Les modeles sont ensuite mis en cache et ne sont plus retelecharges.

---

## Ressources necessaires

| Composant | Minimum recommande |
|-----------|-------------------|
| RAM | 8 Go |
| GPU VRAM | 4 Go (pour executer sur GPU) |
| Stockage | 5 Go libres (cache des modeles) |

> Le pipeline fonctionne sur CPU ou sur un GPU avec au moins 4 Go de VRAM. Une execution GPU prendra quelques secondes.

---

## Installation

### 1. Creer un environnement virtuel Python

```bash
python3 -m venv .venv
```

### 2. Activer l'environnement virtuel

**Sur Linux/macOS :**
```bash
source .venv/bin/activate
```

**Sur Windows (PowerShell) :**
```powershell
.venv\Scripts\Activate.ps1
```

**Sur Windows (CMD) :**
```cmd
.venv\Scripts\activate.bat
```

### 3. Installer les dependances

```bash
pip install -r requirements.txt
```

---

## Utilisation

(Assurez-vous que l'environnement virtuel est activé)

### Lancer le pipeline complet

Placez votre fichier audio a la racine du projet (WAV ou MP3) :

```bash
python main.py --input input.mp3 --output output.wav --device cpu
```

> **Note :** Vous pouvez remplacer `--device cpu` par `--device cuda` pour utiliser votre carte graphique NVIDIA (si elle dispose d'au moins 6 Go de VRAM).

Le terminal affichera le resultat de chaque etape. A la fin, `output.wav` contiendra
la reponse synthetisee par MMS-TTS.

---

### Tester chaque etape independamment

**Tester la transcription (STT) :**
```bash
python stt.py input.mp3 cpu
```
*(Remplacez `cpu` par `cuda` pour tester sur GPU)*

**Tester la generation de texte (LLM) :**
```bash
python llm.py "What is the capital of France?" --device cpu
```

**Tester la synthese vocale (TTS) :**
```bash
python tts.py "Hello, this is a test." output.wav cpu
```
*(Remplacez `cpu` par `cuda` pour tester sur GPU)*

---

## Comment fonctionne MMS-TTS

MMS-TTS est un modele de synthese vocale leger developpe par Meta (Facebook) utilisant l'architecture VITS. Son fonctionnement :

```
Texte d'entree
     |
     v
[MMS-TTS Pipeline] -- genere directement la forme d'onde
     |
     v
Fichier WAV de sortie
```

---

## Comportement attendu

Exemple avec un audio de la question "Quelle est la capitale de la France ?" :

```
==================================================
  PIPELINE : Audio -> Texte -> Réponse -> Audio
==================================================

[1/3] Transcription...
[STT] Chargement du modèle openai/whisper-large-v3 sur cpu...
[STT] Transcription en cours pour : input.mp3
>>> Transcription : Quelle est la capitale de la France ?

[2/3] Génération de la réponse...
[LLM] Chargement du modèle google/flan-t5-base sur cpu...
>>> Réponse : Paris

[3/3] Synthèse vocale...
[TTS] Chargement du modèle facebook/mms-tts-eng sur cpu...
[TTS] Génération vocale pour : Paris
[TTS] Sauvegarde dans : output.wav

==================================================
Pipeline terminé avec succès. Fichier généré : output.wav
==================================================
```
