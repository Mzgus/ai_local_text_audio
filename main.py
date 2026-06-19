import argparse
from stt import transcribe
from llm import generate
from tts import synthesize

def run_pipeline(input_audio: str, output_audio: str, device: str = "cpu"):
    print("=" * 50)
    print("  PIPELINE : Audio -> Texte -> Réponse -> Audio")
    print("=" * 50)

    print("\n[1/3] Transcription...")
    text = transcribe(input_audio, device=device)
    print(f">>> Transcription : {text}\n")

    print("[2/3] Génération de la réponse...")
    reply = generate(text, device=device)
    print(f">>> Réponse : {reply}\n")

    print("[3/3] Synthèse vocale...")
    synthesize(reply, output_audio, device=device)
    
    print("\n" + "=" * 50)
    print(f"Pipeline terminé avec succès. Fichier généré : {output_audio}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline IA local : Audio -> Texte -> Réponse -> Audio")
    parser.add_argument("--input", type=str, default="input.wav")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output, device=args.device)
