FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN gdown "https://drive.google.com/file/d/1Pdo2SjtoEpoIVgRBkti3iFXJrF_5vyCA/view?usp=sharing" -O best_model_phase1.keras && \
    ls -lh best_model_phase1.keras && \
    python -c "import zipfile; zipfile.ZipFile('best_model_phase1.keras')" && \
    echo "✅ Modèle valide !"
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]