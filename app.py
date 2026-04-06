import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

# ============================================
# CONFIG
# ============================================

MODEL_NAME = "best_model_phase1.keras"

# ⚠️ mets ton vrai lien Google Drive ici si besoin
GDRIVE_URL = "https://drive.google.com/uc?id=TON_FILE_ID"

# ============================================
# DEBUG
# ============================================

st.write("📁 Fichiers disponibles :", os.listdir("."))

# ============================================
# DOWNLOAD SI MANQUANT
# ============================================

if not os.path.exists(MODEL_NAME):
    st.warning("⚠️ Modèle introuvable, téléchargement...")

    try:
        gdown.download(GDRIVE_URL, MODEL_NAME, quiet=False)
        st.success("✅ Modèle téléchargé")
    except Exception as e:
        st.error("❌ Erreur téléchargement : " + str(e))

# ============================================
# LOAD MODEL (SECURISE)
# ============================================

@st.cache_resource
def load_fruit_model():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, MODEL_NAME)

        if not os.path.exists(model_path):
            st.error("❌ Modèle toujours introuvable après vérification")
            return None

        st.write("📍 Chargement depuis :", model_path)

        model = load_model(model_path)
        st.success("✅ Modèle chargé")

        return model

    except Exception as e:
        st.error("❌ Erreur chargement : " + str(e))
        return None

# ============================================
# UI
# ============================================

st.title("🍎 Fruit Classifier")

uploaded = st.file_uploader("📤 Upload une image", type=["jpg", "png", "jpeg"])

model = None

# 👉 IMPORTANT : charger seulement ici
if uploaded:
    if model is None:
        model = load_fruit_model()

    if model is not None:
        try:
            img = Image.open(uploaded).resize((224, 224))
            st.image(img, caption="📸 Image")

            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)

            st.write("🔢 Résultat :", prediction)

            predicted_class = np.argmax(prediction)

            st.success(f"✅ Classe : {predicted_class}")

        except Exception as e:
            st.error("❌ Erreur prédiction : " + str(e))