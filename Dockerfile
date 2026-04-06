FROM python:3.11-slim
WORKDIR /app

# Dépendances système pour éviter les crashs d'image
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le lien Drive direct avec option --confirm
# Utilisation de l'URL directe de téléchargement au lieu de l'URL de vue
RUN gdown "https://drive.google.com/uc?id=1Pdo2SjtoEpoIVgRBkti3iFXJrF_5vyCA" -O /app/best_model_phase1.keras && \
    ls -lh /app/best_model_phase1.keras
# Vérification immédiate dans les logs Railway
RUN ls -lh best_model_phase1.keras

COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]