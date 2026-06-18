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
|-- tts.py            <- Etape 3 : Texte -> Audio (orpheus-3b-0.1-ft)
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
| TTS | `canopylabs/orpheus-3b-0.1-ft` | ~6 Go | Synthese vocale tres naturelle (Llama 3B + codec SNAC) |

> **Note importante** : Le total a telecharger est environ **9 Go** lors du premier lancement.
> Les modeles sont ensuite mis en cache et ne sont plus retelecharges.

---

## Ressources necessaires

| Composant | Minimum recommande |
|-----------|-------------------|
| RAM | 16 Go (sans GPU) |
| GPU VRAM | 12 Go (pour executer sur GPU) |
| Stockage | 12 Go libres (cache des modeles) |

> Si vous n'avez pas de GPU, le pipeline fonctionne sur CPU mais sera **tres lent**
> (plusieurs minutes par etape). Avec un GPU NVIDIA, chaque etape prend quelques secondes.

---

## Installation

### 1. Creer un environnement virtuel Python

```powershell
python -m venv .venv
```

### 2. Installer les dependances

```powershell
.venv\Scripts\pip install -r requirements.txt
```

---

## Utilisation

### Lancer le pipeline complet

Placez votre fichier audio a la racine du projet (WAV ou MP3) :

```powershell
.venv\Scripts\python main.py --input input.mp3 --output output.wav
```

Le terminal affichera le resultat de chaque etape. A la fin, `output.wav` contiendra
la reponse synthetisee par Orpheus.

---

### Tester chaque etape independamment

**Tester la transcription (STT) :**
```powershell
.venv\Scripts\python stt.py input.mp3
```

**Tester la generation de texte (LLM) :**
```powershell
.venv\Scripts\python llm.py "What is the capital of France?"
```

**Tester la synthese vocale (TTS) :**
```powershell
.venv\Scripts\python tts.py "Hello, this is a test." output.wav tara
```

Voix disponibles pour Orpheus : `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`

---

## Comment fonctionne Orpheus TTS

Orpheus n'est pas un modele TTS classique. C'est un LLM (grand modele de langage)
base sur Llama 3B, specialise pour la synthese vocale. Son fonctionnement :

```
Texte d'entree
     |
     v
[Orpheus LLM] -- genere des tokens audio (codes SNAC)
     |
     v
[Codec SNAC]  -- decode les tokens en signal audio
     |
     v
Fichier WAV de sortie
```

Le codec **SNAC** (Simple Neural Audio Codec) travaille sur 3 couches de resolution
pour reconstruire un audio de qualite a 24 000 Hz.

---

## Comportement attendu

Exemple avec un audio de la question "Quelle est la capitale de la France ?" :

```
==================================================
  PIPELINE : Audio -> Texte -> Reponse -> Audio
==================================================

[1/3] Transcription de l'audio en texte...
[STT] Chargement du modele openai/whisper-large-v3...
>>> Transcription : Quelle est la capitale de la France ?

[2/3] Generation d'une reponse texte...
>>> Reponse : Paris

[3/3] Synthese vocale de la reponse...
[TTS] Generation audio avec la voix 'tara'...
>>> Fichier audio genere : output.wav

==================================================
  Pipeline termine avec succes !
==================================================
```
