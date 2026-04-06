import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import plotly.graph_objects as go
import time

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
model_path = 'best_model_phase1.keras'
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
#  CLASSES ET INFOS
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
#  CHARGEMENT MODÈLE
# ─────────────────────────────────────────────
@st.cache_resource
def load_fruit_model():
    if not os.path.exists(model_path):
        st.error(f"❌ Modèle introuvable : {model_path}")
        st.write("📁 Fichiers dans /app :", os.listdir('/app'))
        st.stop()
    return load_model(model_path)

model = load_fruit_model()

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
    idx   = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx]) * 100, preds

# ─────────────────────────────────────────────
#  INTERFACE
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">FreshScan</p>
    <p class="main-subtitle">Détecteur de fraîcheur — IA MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <div class="info-icon">🧠</div>
        <div class="info-title">Modèle</div>
        <div class="info-value">MobileNetV2</div>
    </div>
    <div class="info-card">
        <div class="info-icon">🎯</div>
        <div class="info-title">Précision</div>
        <div class="info-value">99.86%</div>
    </div>
    <div class="info-card">
        <div class="info-icon">🍓</div>
        <div class="info-title">Classes</div>
        <div class="info-value">6 types</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Dépose une photo de fruit ici",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible"
)

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.image(img_pil, use_container_width=True, caption="Photo analysée")

    with col2:
        with st.spinner("Analyse en cours..."):
            time.sleep(0.5)
            classe, confiance, all_preds = predict(img_pil, model)

        info       = CLASS_INFO[classe]
        comestible = 'rotten' not in classe
        card_cls   = 'comestible' if comestible else 'non-comestible'
        verdict    = '✅ COMESTIBLE' if comestible else '❌ NON COMESTIBLE'

        st.markdown(f"""
        <div class="result-card {card_cls}">
            <span class="verdict-emoji">{info['emoji']}</span>
            <div class="verdict-text {card_cls}">{verdict}</div>
            <div class="class-badge">{info['nom'].upper()}</div>
            <div class="confidence-text">
                Confiance : <span class="confidence-value">{confiance:.1f}%</span>
            </div>
            <div style="margin-top:1rem; color:#888; font-size:0.85rem;">
                💡 {info['conseil']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📊 Probabilités par classe")

    fig = go.Figure(go.Bar(
        x=[p * 100 for p in all_preds],
        y=CLASS_NAMES,
        orientation='h',
        marker=dict(
            color=[BAR_COLORS[c] for c in CLASS_NAMES],
            opacity=0.85,
            line=dict(width=0)
        ),
        text=[f"{p*100:.1f}%" for p in all_preds],
        textposition='outside',
        textfont=dict(color='#888', size=12)
    ))

    fig.update_layout(
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='#13131a',
        font=dict(family='DM Sans', color='#888'),
        xaxis=dict(
            showgrid=True, gridcolor='#1e1e2e',
            range=[0, 110], ticksuffix='%', color='#555'
        ),
        yaxis=dict(color='#aaa', tickfont=dict(size=13)),
        margin=dict(l=10, r=40, t=20, b=20),
        height=280,
        showlegend=False,
        bargap=0.35
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Analyser une autre photo"):
        st.rerun()

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #333;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">🍎🍌🍊</div>
        <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; color: #444;">
            Charge une photo pour commencer l'analyse
        </div>
        <div style="font-size: 0.85rem; color: #333; margin-top: 0.5rem;">
            Formats acceptés : JPG, PNG, WEBP
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    FRESHSCAN · MOBILENETV2 · TRANSFER LEARNING · 99.86% ACCURACY
</div>
""", unsafe_allow_html=True)