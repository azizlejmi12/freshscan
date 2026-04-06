FROM python:3.11-slim
WORKDIR /app

# Installation des dépendances système nécessaires pour le traitement d'image
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Téléchargement propre du modèle
RUN gdown --confirm "1Pdo2SjtoEpoIVgRBkti3iFXJrF_5vyCA" -O /app/best_model_phase1.keras

# Copie du reste du code
COPY app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]