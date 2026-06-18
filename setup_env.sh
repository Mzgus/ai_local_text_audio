#!/bin/bash

# Configuration d'arrêt en cas d'erreur
set -e

echo "=== Création de l'environnement virtuel Python ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Environnement virtuel créé dans .venv."
else
    echo "L'environnement virtuel existe déjà."
fi

echo "=== Activation de l'environnement virtuel et mise à jour de pip ==="
source .venv/bin/activate

pip install --upgrade pip

echo "=== Installation des dépendances depuis requirements.txt ==="
pip install -r requirements.txt

echo "=== Installation terminée ==="
echo "Pour activer l'environnement : source .venv/bin/activate"
