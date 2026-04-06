FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Télécharger le modèle au moment du BUILD (pas au runtime)
RUN pip install gdown && \
    gdown "1E6AQE-DnggKsVgMI-4oKBm5cCCqWMyXh" -O best_model_phase1.h5
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]