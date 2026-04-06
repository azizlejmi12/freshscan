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
GDRIVE_URL = "https://drive.google.com/file/d/1Pdo2SjtoEpoIVgRBkti3iFXJrF_5vyCA/view?usp=sharing"

# ============================================
# DEBUG (IMPORTANT POUR RAILWAY)
# ============================================

st.write("📁 Fichiers dans le dossier :", os.listdir("."))
st.write("📁 Chemin absolu :", os.path.abspath("."))

# ============================================
# TELECHARGEMENT SI MANQUANT (avec conversion ID)
# ============================================

def download_from_gdrive(url, output):
    """Télécharge depuis Google Drive en gérant les fichiers larges"""
    try:
        # Convertir lien de partage en lien de téléchargement direct
        if "drive.google.com/file/d/" in url:
            file_id = url.split("/d/")[1].split("/")[0]
            direct_url = f"https://drive.google.com/uc?id={file_id}"
        else:
            direct_url = url
            
        gdown.download(direct_url, output, quiet=False)
        return True
    except Exception as e:
        st.error(f"❌ Erreur téléchargement : {str(e)}")
        return False

if not os.path.exists(MODEL_NAME):
    st.warning(f"⚠️ {MODEL_NAME} introuvable, téléchargement en cours...")
    
    if download_from_gdrive(GDRIVE_URL, MODEL_NAME):
        st.success("✅ Modèle téléchargé avec succès")
    else:
        st.error("❌ Échec du téléchargement")

# ============================================
# VERIFICATION FICHIER
# ============================================

if os.path.exists(MODEL_NAME):
    st.write("✅ Fichier trouvé")
    st.write("📦 Taille :", os.path.getsize(MODEL_NAME), "bytes")
else:
    st.error("❌ Fichier toujours introuvable !")

# ============================================
# LOAD MODEL (CORRIGÉ)
# ============================================

@st.cache_resource
def load_fruit_model():
    # Utiliser le chemin relatif simple (WORKDIR /app dans Docker)
    model_path = MODEL_NAME  # Pas besoin de chemin absolu complexe
    
    st.write("📍 Chargement modèle depuis :", os.path.abspath(model_path))
    
    if not os.path.exists(model_path):
        st.error(f"❌ Modèle introuvable : {model_path}")
        # Liste tous les fichiers pour debug
        st.write("📁 Fichiers disponibles :", os.listdir("."))
        return None
    
    try:
        model = load_model(model_path)
        st.success("✅ Modèle chargé avec succès")
        return model
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle : {str(e)}")
        return None

# ============================================
# INTERFACE
# ============================================

st.title("🍎 Fruit Classifier")

uploaded = st.file_uploader("📤 Upload une image", type=["jpg", "png", "jpeg"])

if uploaded:
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
    else:
        st.error("❌ Impossible de charger le modèle")