# llm.py - Etape 2 : Generation de texte (Text-to-Text)
# -------------------------------------------------------
# Ce script prend un texte en entree et genere un nouveau texte.
# Modele utilise : google/flan-t5-base
#   -> Un petit modele de langage capable de repondre a des questions
#      et de suivre des instructions en anglais.
#
# Note : Flan-T5 fonctionne mieux en anglais. Si votre audio est en
# francais, la reponse sera en anglais. Vous pouvez changer de modele
# dans les constantes ci-dessous.

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig

# --- Constantes ---
LLM_MODEL = "google/flan-t5-base"


def generate(texte_entree: str, longueur_max: int = 128, device: str = "cpu") -> str:
    """
    Genere une reponse textuelle a partir d'un texte d'entree.

    Parametres :
        texte_entree  : le texte a traiter (ex: une transcription)
        longueur_max  : longueur maximale de la reponse generee
        device        : appareil cible ("cpu" ou "cuda")

    Retourne :
        La reponse generee par le modele.
    """
    print(f"[LLM] Chargement du modele ({LLM_MODEL}) sur {device}...")

    # On charge la configuration du modele pour savoir comment le charger
    config = AutoConfig.from_pretrained(LLM_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    # Les modeles "encoder-decoder" (comme T5) se chargent differemment
    # des modeles "generatifs" classiques (comme LLaMA, Mistral, etc.)
    if getattr(config, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).to(device)
        # Flan-T5 est deja entraine sur des instructions (par ex: "Tell me a joke."),
        # on lui passe donc le texte d'entree directement pour eviter de l'induire en erreur.
        prompt = texte_entree
    else:
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL).to(device)
        prompt = texte_entree

    print(f"[LLM] Generation d'une reponse pour : {texte_entree}")

    # Conversion du texte en tokens (format compris par le modele)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generation sans calcul de gradient pour economiser la memoire
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=longueur_max,
            do_sample=True,   # Active l'echantillonnage (reponses moins repetitives)
            temperature=0.3,  # Temperature basse pour eviter les reponses incoherentes
            top_k=50,         # Limite aux 50 tokens les plus probables
            top_p=0.95,       # Filtrage par probabilite cumulee
        )

    # Decodage des tokens en texte lisible
    reponse = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Pour les modeles generatifs, la reponse peut contenir le prompt -- on l'enleve
    if not getattr(config, "is_encoder_decoder", False) and reponse.startswith(prompt):
        reponse = reponse[len(prompt):].strip()

    print(f"[LLM] Reponse generee : {reponse}")
    return reponse


# --- Test autonome : lancer ce fichier seul pour tester la generation ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test autonome LLM")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="What is the capital of France?",
        help="Le texte a envoyer au LLM"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Appareil cible ('cpu' ou 'cuda')"
    )
    args = parser.parse_args()

    reponse = generate(args.prompt, device=args.device)
    print("\n=== Resultat ===")
    print(reponse)
