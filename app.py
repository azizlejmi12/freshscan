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

# 👉 (OPTIONNEL) lien Google Drive si problème Railway
# remplace par ton lien si nécessaire
GDRIVE_URL = "https://drive.google.com/file/d/1Pdo2SjtoEpoIVgRBkti3iFXJrF_5vyCA/view?usp=sharing"

# ============================================
# DEBUG (IMPORTANT POUR RAILWAY)
# ============================================

st.write("📁 Fichiers dans le dossier :", os.listdir("."))

# ============================================
# TELECHARGEMENT SI MANQUANT
# ============================================

if not os.path.exists(MODEL_NAME):
    st.warning("⚠️ Modèle introuvable, téléchargement en cours...")
    
    try:
        gdown.download(GDRIVE_URL, MODEL_NAME, quiet=False)
        st.success("✅ Modèle téléchargé avec succès")
    except Exception as e:
        st.error("❌ Erreur téléchargement modèle : " + str(e))

# ============================================
# VERIFICATION FICHIER
# ============================================

if os.path.exists(MODEL_NAME):
    st.write("✅ Fichier trouvé")
    st.write("📦 Taille :", os.path.getsize(MODEL_NAME), "bytes")
else:
    st.error("❌ Fichier toujours introuvable !")

# ============================================
# LOAD MODEL (OPTIMISÉ)
# ============================================

@st.cache_resource
def load_fruit_model():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, MODEL_NAME)
        
        st.write("📍 Chargement modèle depuis :", model_path)
        
        model = load_model(model_path)
        st.success("✅ Modèle chargé avec succès")
        return model
    
    except Exception as e:
        st.error("❌ Erreur chargement modèle : " + str(e))
        return None

# ============================================
# INTERFACE
# ============================================

st.title("🍎 Fruit Classifier")

uploaded = st.file_uploader("📤 Upload une image", type=["jpg", "png", "jpeg"])

model = None

if uploaded:
    # Charger modèle seulement quand nécessaire
    if model is None:
        model = load_fruit_model()

    if model is not None:
        try:
            img = Image.open(uploaded).resize((224, 224))
            st.image(img, caption="📸 Image chargée")

            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)

            st.write("🔢 Résultat brut :", prediction)

            predicted_class = np.argmax(prediction)

            st.success(f"✅ Classe prédite : {predicted_class}")

        except Exception as e:
            st.error("❌ Erreur prédiction : " + str(e))