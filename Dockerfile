FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN gdown "1Pdo2SjtoEpoIVgRBkti3iFXJrF_5vyCA" -O /app/best_model_phase1.keras && \
    ls -lh /app/best_model_phase1.keras && \
    echo "✅ Modèle téléchargé !"
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]