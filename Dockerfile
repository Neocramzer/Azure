FROM python:3.9-slim-bookworm

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python existantes
RUN pip install --no-cache-dir "flask<3" "pillow<11" "numpy<2" tflite-runtime~=2.13.0

# Installation des dépendances pour le modèle AQI simple
RUN pip install --no-cache-dir \
    pandas==1.5.3 \
    scikit-learn==1.5.1 \
    joblib==1.3.2

# Copie de l'application
COPY app /app

# Copie du dataset CSV pour entraînement
COPY air_pollution_data.csv /app/

# Configuration du port
EXPOSE 80

# Répertoire de travail
WORKDIR /app

# Commande de démarrage
CMD ["python", "-u", "app.py"]
