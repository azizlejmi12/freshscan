import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import plotly.graph_objects as go
import time

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FreshScan — Détecteur de fruits",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #f0ede8; }
.main-header { text-align: center; padding: 2.5rem 0 1rem 0; }
.main-title {
    font-family: 'Syne', sans-serif; font-size: 3.2rem; font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #a8ff78, #78ffd6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; line-height: 1;
}
.main-subtitle {
    font-family: 'DM Sans', sans-serif; font-size: 1rem; color: #888;
    margin-top: 0.5rem; font-weight: 300; letter-spacing: 2px; text-transform: uppercase;
}
.result-card { border-radius: 20px; padding: 2rem; margin: 1.5rem 0; text-align: center; }
.result-card.comestible { background: linear-gradient(135deg, #0d2b1a, #0a1f12); border: 1.5px solid #2ecc71; }
.result-card.non-comestible { background: linear-gradient(135deg, #2b0d0d, #1f0a0a); border: 1.5px solid #e74c3c; }
.verdict-emoji { font-size: 4rem; margin-bottom: 0.5rem; display: block; }
.verdict-text { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; letter-spacing: -1px; }
.verdict-text.comestible { color: #2ecc71; }
.verdict-text.non-comestible { color: #e74c3c; }
.class-badge {
    display: inline-block; background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15); border-radius: 50px;
    padding: 0.4rem 1.2rem; font-size: 0.9rem; color: #ccc;
    margin-top: 0.8rem; letter-spacing: 1px; font-weight: 500;
}
.confidence-text { font-family: 'Syne', sans-serif; font-size: 1.1rem; color: #888; margin-top: 0.5rem; }
.confidence-value { color: #f0ede8; font-weight: 700; }
.info-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }
.info-card { background: #13131a; border: 1px solid #1e1e2e; border-radius: 14px; padding: 1.2rem; text-align: center; }
.info-icon { font-size: 1.8rem; }
.info-title { font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.4rem; }
.info-value { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: #f0ede8; margin-top: 0.2rem; }
.stButton > button {
    background: linear-gradient(135deg, #a8ff78, #78ffd6) !important;
    color: #0a0a0f !important; border: none !important; border-radius: 50px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.95rem !important; padding: 0.6rem 2rem !important;
    letter-spacing: 1px !important; width: 100% !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
[data-testid="stFileUploader"] {
    background: #13131a !important; border-radius: 16px !important;
    border: 1.5px dashed #2a2a3a !important; padding: 1rem !important;
}
.divider { height: 1px; background: linear-gradient(90deg, transparent, #2a2a3a, transparent); margin: 2rem 0; }
.footer { text-align: center; color: #333; font-size: 0.8rem; padding: 2rem 0 1rem 0; letter-spacing: 1px; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CLASSES
# ─────────────────────────────────────────────
CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshoranges',
    'rottenapples', 'rottenbanana', 'rottenoranges'
]

CLASS_INFO = {
    'freshapples':   {'emoji': '🍎', 'nom': 'Pomme fraîche',  'conseil': 'Parfaite à consommer !'},
    'freshbanana':   {'emoji': '🍌', 'nom': 'Banane fraîche', 'conseil': 'Idéale pour manger maintenant.'},
    'freshoranges':  {'emoji': '🍊', 'nom': 'Orange fraîche', 'conseil': 'Pleine de vitamines C !'},
    'rottenapples':  {'emoji': '🍎', 'nom': 'Pomme pourrie',  'conseil': 'À jeter immédiatement.'},
    'rottenbanana':  {'emoji': '🍌', 'nom': 'Banane pourrie', 'conseil': 'Peut servir pour un gâteau.'},
    'rottenoranges': {'emoji': '🍊', 'nom': 'Orange pourrie', 'conseil': 'Ne pas consommer.'},
}

BAR_COLORS = {
    'freshapples':   '#2ecc71',
    'freshbanana':   '#a8ff78',
    'freshoranges':  '#f39c12',
    'rottenapples':  '#e74c3c',
    'rottenbanana':  '#c0392b',
    'rottenoranges': '#e67e22',
}

# ─────────────────────────────────────────────
#  CHARGEMENT MODÈLE (.keras)
# ─────────────────────────────────────────────
@st.cache_resource
def load_fruit_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "best_model_phase1.keras")

    st.write("📂 Dossier :", BASE_DIR)
    st.write("📁 Fichiers :", os.listdir(BASE_DIR))

    if not os.path.exists(model_path):
        st.error(f"❌ Modèle introuvable : {model_path}")
        st.stop()

    try:
        model = load_model(model_path)
        st.success("✅ Modèle chargé avec succès")
        return model
    except Exception as e:
        st.error("❌ Erreur chargement modèle")
        st.write(e)
        st.stop()

# ─────────────────────────────────────────────
#  PRÉDICTION
# ─────────────────────────────────────────────
def predict(img_pil, model):
    img = img_pil.resize((224, 224))
    arr = np.array(img) / 255.0

    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]

    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx]) * 100, preds

# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">FreshScan</p>
    <p class="main-subtitle">Détecteur de fraîcheur — IA MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Dépose une photo de fruit ici",
    type=["jpg", "jpeg", "png", "webp"]
)

model = None

if uploaded:
    if model is None:
        model = load_fruit_model()

    img_pil = Image.open(uploaded).convert("RGB")

    st.image(img_pil, use_container_width=True)

    with st.spinner("Analyse en cours..."):
        time.sleep(0.5)
        classe, confiance, all_preds = predict(img_pil, model)

    info = CLASS_INFO[classe]
    st.success(f"{info['emoji']} {info['nom']} — {confiance:.2f}%")

    fig = go.Figure(go.Bar(
        x=[p * 100 for p in all_preds],
        y=CLASS_NAMES,
        orientation='h'
    ))

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Charge une image pour commencer")