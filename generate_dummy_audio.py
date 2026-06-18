import numpy as np
import scipy.io.wavfile
import config

def create_sine_wave(filename="input.wav", duration=3, frequency=440):
    """Génère un fichier audio WAV simple contenant une onde sinusoïdale de 440Hz."""
    t = np.linspace(0, duration, int(config.SAMPLE_RATE * duration), endpoint=False)
    # Génération d'une onde sinusoïdale simple
    data = np.sin(2 * np.pi * frequency * t)
    # Normalisation
    data = data / np.max(np.abs(data))
    # Sauvegarde au format WAV 16kHz
    scipy.io.wavfile.write(filename, config.SAMPLE_RATE, data.astype(np.float32))
    print(f"Fichier audio de test généré : {filename}")

if __name__ == "__main__":
    create_sine_wave()
