FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On copie tout, mais on vérifie spécifiquement le fichier .h5
COPY . .

# Cette ligne va échouer au build si le fichier n'est pas là, 
# ça évite de perdre du temps avec un déploiement vide.
RUN ls -lh best_model_phase1.h5

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]