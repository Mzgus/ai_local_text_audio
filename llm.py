import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

LLM_MODEL = "google/flan-t5-base"

def generate(texte_entree: str, longueur_max: int = 128, device: str = "cpu") -> str:
    """
    Génère une réponse textuelle avec google/flan-t5-base.
    """
    print(f"[LLM] Chargement du modèle {LLM_MODEL} sur {device}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).to(device)
    
    inputs = tokenizer(texte_entree, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=longueur_max,
            do_sample=True,
            temperature=0.3,
            top_k=50,
            top_p=0.95
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test autonome LLM")
    parser.add_argument("prompt", type=str, nargs="?", default="What is the capital of France?")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    reponse = generate(args.prompt, device=args.device)
    print(f"\n=== Résultat ===\n{reponse}")
