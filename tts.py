# tts.py - Etape 3 : Synthese Vocale Texte -> Audio (Text-to-Speech)
# -------------------------------------------------------------------
# Modele utilise : canopylabs/orpheus-3b-0.1-ft
#
# Orpheus est un modele LLM base sur Llama 3B specialise dans la synthese vocale.
# Il ne genere pas directement du WAV : il produit des "tokens audio" (codes SNAC)
# qu'il faut ensuite decoder avec le codec SNAC pour obtenir un fichier audio.
#
# Pipeline de generation :
#   Texte -> [Orpheus LLM] -> Tokens audio SNAC -> [SNAC Codec] -> Audio WAV
#
# ATTENTION : Necesssite ~6 Go de RAM minimum (modele 3 milliards de parametres).
#             Sur CPU c'est tres lent (plusieurs minutes). GPU fortement recommande.
#
# Installation requise (en plus de requirements.txt) :
#   pip install snac

import torch
import numpy as np
import scipy.io.wavfile
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Constantes ---
TTS_MODEL = "canopylabs/orpheus-3b-0.1-ft"
SNAC_MODEL = "hubertsiuzdak/snac_24khz"
DEFAULT_VOICE = "tara"   # Voix disponibles : tara, leah, jess, leo, dan, mia, zac, zoe
SAMPLE_RATE = 24000      # Le codec SNAC 24khz produit de l'audio a 24 000 Hz

# ID de debut des tokens audio dans le vocabulaire du modele
# Les tokens audio commencent a l'index 128266 dans le vocabulaire d'Orpheus
AUDIO_TOKEN_OFFSET = 128266


def _decode_snac_tokens(audio_token_ids: list, snac_model) -> np.ndarray:
    """
    Decode les tokens audio generes par Orpheus en signal audio WAV.

    Orpheus genere des tokens organises en groupes de 7 pour le codec SNAC :
      - 1 token pour la couche 0 (basse resolution)
      - 2 tokens pour la couche 1 (resolution moyenne)
      - 4 tokens pour la couche 2 (haute resolution)

    Parametre :
        audio_token_ids : liste des IDs de tokens audio (deja filtres, mais bruts avec offsets)
        snac_model      : le modele SNAC charge

    Retourne :
        Tableau numpy contenant la forme d'onde audio.
    """
    # Nombre de frames completes (groupes de 7 tokens)
    n_frames = len(audio_token_ids) // 7
    if n_frames == 0:
        print("[TTS] Avertissement : aucun token audio genere.")
        return np.zeros(SAMPLE_RATE, dtype=np.float32)

    # On garde seulement les frames completes
    tokens = audio_token_ids[:n_frames * 7]

    # Reorganisation des tokens dans les 3 couches du codec SNAC avec soustraction des offsets respectifs
    codes_0, codes_1, codes_2 = [], [], []
    for i in range(n_frames):
        frame = tokens[i * 7: (i + 1) * 7]
        
        # Position 0 -> Layer 0 (coarse): offset 128266
        c0 = frame[0] - 128266
        codes_0.append(max(0, min(4095, c0)))
        
        # Position 1 -> Layer 1 (medium): offset 128266 + 4096 = 132362
        c1_a = frame[1] - 132362
        # Position 4 -> Layer 1 (medium): offset 128266 + 4 * 4096 = 144650
        c1_b = frame[4] - 144650
        codes_1.extend([max(0, min(4095, c1_a)), max(0, min(4095, c1_b))])
        
        # Position 2 -> Layer 2 (fine): offset 128266 + 2 * 4096 = 136458
        c2_a = frame[2] - 136458
        # Position 3 -> Layer 2 (fine): offset 128266 + 3 * 4096 = 140554
        c2_b = frame[3] - 140554
        # Position 5 -> Layer 2 (fine): offset 128266 + 5 * 4096 = 148746
        c2_c = frame[5] - 148746
        # Position 6 -> Layer 2 (fine): offset 128266 + 6 * 4096 = 152842
        c2_d = frame[6] - 152842
        codes_2.extend([
            max(0, min(4095, c2_a)),
            max(0, min(4095, c2_b)),
            max(0, min(4095, c2_c)),
            max(0, min(4095, c2_d))
        ])

    device = next(snac_model.parameters()).device
    codes = [
        torch.tensor(codes_0, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(codes_1, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(codes_2, dtype=torch.long, device=device).unsqueeze(0),
    ]

    with torch.no_grad():
        audio_tensor = snac_model.decode(codes)

    return audio_tensor.squeeze().cpu().numpy()


def synthesize(texte: str, output_path: str, voice: str = DEFAULT_VOICE) -> None:
    """
    Convertit un texte en fichier audio WAV avec la voix Orpheus.

    Parametres :
        texte       : le texte a prononcer
        output_path : chemin du fichier audio de sortie (ex: "output.wav")
        voice       : voix a utiliser (tara, leah, jess, leo, dan, mia, zac, zoe)
    """
    if not texte:
        raise ValueError("Le texte a synthetiser est vide.")

    # Nettoyage du texte pour eviter les caracteres speciaux (guillemets courbes, apostrophes typographiques, etc.)
    # qui peuvent perturber le tokenizer du modele de synthese vocale
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
    for orig, rep in replacements.items():
        texte = texte.replace(orig, rep)
    texte = " ".join(texte.split())

    # Import de snac avec un message d'erreur utile si non installe
    try:
        from snac import SNAC
    except ImportError:
        raise ImportError(
            "La bibliotheque 'snac' est requise pour Orpheus TTS.\n"
            "Installez-la avec : pip install snac"
        )

    print(f"[TTS] Chargement du modele Orpheus ({TTS_MODEL})...")
    print("[TTS] (Premier lancement : telechargement ~6 Go, patience...)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TTS_MODEL,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    print(f"[TTS] Chargement du codec SNAC ({SNAC_MODEL})...")
    snac = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)

    # Orpheus est un LLM Llama 3B finetuné pour le TTS.
    # Le format attendu pour la génération est : "{voix}: {texte}"
    # suivi directement du token déclencheur audio (AUDIO_START_TOKEN_ID)
    prompt = f"{voice}: {texte}"
    encoding = tokenizer(prompt, return_tensors="pt")

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Token spécial qui déclenche la génération audio dans Orpheus (ID 128259)
    # Ce token signifie "maintenant génère de l'audio" dans le vocabulaire étendu
    AUDIO_START_TOKEN_ID = 128259
    audio_trigger = torch.tensor([[AUDIO_START_TOKEN_ID]], device=device)
    input_ids = torch.cat([input_ids, audio_trigger], dim=1)

    # Mettre à jour l'attention mask pour inclure le token déclencheur
    attention_trigger = torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=device)
    attention_mask = torch.cat([attention_mask, attention_trigger], dim=1)

    print(f"[TTS] Generation audio avec la voix '{voice}' pour : {texte}")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,      # Assez de tokens pour quelques secondes d'audio
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,   # Requis par Orpheus pour éviter les répétitions
            eos_token_id=128258,      # Token de fin d'audio dans le vocabulaire Orpheus
        )

    # On extrait seulement les tokens générés (pas le prompt)
    generated_ids = output_ids[0][input_ids.shape[1]:].tolist()

    print(f"[TTS] Tokens generes (total) : {len(generated_ids)}")
    if generated_ids:
        print(f"[TTS] Plage des IDs : {min(generated_ids)} -> {max(generated_ids)}")

    # On filtre les tokens audio (IDs >= AUDIO_TOKEN_OFFSET)
    # Note : On conserve les IDs bruts car les offsets par couche (0, 1, 2) sont retires lors du decodage
    audio_token_ids = [
        t
        for t in generated_ids
        if t >= AUDIO_TOKEN_OFFSET
    ]

    print(f"[TTS] {len(audio_token_ids)} tokens audio filtres, decodage SNAC...")

    # Décodage des tokens audio en signal audio
    waveform = _decode_snac_tokens(audio_token_ids, snac)

    # Normalisation pour éviter la saturation audio
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val

    # Sauvegarde au format WAV
    print(f"[TTS] Sauvegarde de l'audio dans : {output_path}")
    scipy.io.wavfile.write(output_path, rate=SAMPLE_RATE, data=waveform.astype(np.float32))
    print(f"[TTS] Fichier audio genere avec succes ({SAMPLE_RATE} Hz).")


# --- Test autonome : lancer ce fichier seul pour tester la synthese ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python tts.py \"<texte>\" [fichier_sortie.wav] [voix]")
        print("Exemple : python tts.py \"Hello world\" output.wav tara")
        print("Voix disponibles : tara, leah, jess, leo, dan, mia, zac, zoe")
        sys.exit(1)

    texte = sys.argv[1]
    fichier_sortie = sys.argv[2] if len(sys.argv) > 2 else "output.wav"
    voix = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_VOICE

    synthesize(texte, fichier_sortie, voix)
    print(f"\n=== Resultat : fichier '{fichier_sortie}' cree avec la voix '{voix}' ===")
