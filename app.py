import os
import time
import importlib

import av
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from keras.models import load_model

try:
    streamlit_webrtc = importlib.import_module("streamlit_webrtc")
    VideoProcessorBase = streamlit_webrtc.VideoProcessorBase
    WebRtcMode = streamlit_webrtc.WebRtcMode
    webrtc_streamer = streamlit_webrtc.webrtc_streamer
    HAS_WEBRTC = True
except ImportError:
    VideoProcessorBase = object
    HAS_WEBRTC = False

try:
    st_autorefresh = importlib.import_module("streamlit_autorefresh").st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

MODEL_CANDIDATES = [
    "/app/best_model_phase1.keras",
    "best_model_phase1.keras",
]

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]

CLASS_INFO = {
    "freshapples": {"emoji": "🍎", "nom": "Pomme fraiche", "conseil": "Parfaite a consommer !"},
    "freshbanana": {"emoji": "🍌", "nom": "Banane fraiche", "conseil": "Ideale pour manger maintenant."},
    "freshoranges": {"emoji": "🍊", "nom": "Orange fraiche", "conseil": "Pleine de vitamines C !"},
    "rottenapples": {"emoji": "🍎", "nom": "Pomme pourrie", "conseil": "A jeter immediatement."},
    "rottenbanana": {"emoji": "🍌", "nom": "Banane pourrie", "conseil": "Peut servir pour un gateau."},
    "rottenoranges": {"emoji": "🍊", "nom": "Orange pourrie", "conseil": "Ne pas consommer."},
}

BAR_COLORS = {
    "freshapples": "#2ecc71",
    "freshbanana": "#a8ff78",
    "freshoranges": "#f39c12",
    "rottenapples": "#e74c3c",
    "rottenbanana": "#c0392b",
    "rottenoranges": "#e67e22",
}

DISEASE_ANALYSIS = {
    "rottenapples": {
        "titre": "Analyse de maladie probable",
        "symptomes": ["Moisissure possible", "Tissu ramolli", "Decomposition avancee"],
        "cause": "Stockage prolonge, humidite ou choc sur le fruit.",
        "conseil": "Jeter le fruit et verifier les autres pommes a proximite.",
    },
    "rottenbanana": {
        "titre": "Analyse de degradation probable",
        "symptomes": ["Noircissement de la peau", "Fermentation", "Deshydratation"],
        "cause": "Surmaturite, chaleur ou exposition a l'air.",
        "conseil": "Si l'odeur est forte ou la chair molle, ne pas consommer.",
    },
    "rottenoranges": {
        "titre": "Analyse de contamination probable",
        "symptomes": ["Moisissure superficielle", "Taches sombres", "Ramollissement"],
        "cause": "Humidite excessive ou conservation trop longue.",
        "conseil": "Ecarter le fruit et nettoyer la zone de stockage.",
    },
}

LOCAL_DETECTION_MIN_AREA_RATIO = 0.015
LOCAL_DETECTION_MAX_BOXES = 4
LOCAL_MAX_DETECT_SIDE = 320
LOCAL_MODEL_MIN_CONFIDENCE = 58.0
LOCAL_CROP_MARGIN_RATIO = 0.12


st.set_page_config(
    page_title="FreshScan - Detecteur de fruits",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


def resolve_model_path() -> str | None:
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


@st.cache_resource
def load_fruit_model():
    model_path = resolve_model_path()
    if model_path is None:
        return None, None
    return load_model(model_path), model_path


def predict(img_pil: Image.Image, model):
    img = img_pil.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]) * 100.0, preds


def predict_robust(img_pil: Image.Image, model):
    rgb = img_pil.convert("RGB")
    w, h = rgb.size
    side = min(w, h)

    cx = (w - side) // 2
    cy = (h - side) // 2
    center_crop = rgb.crop((cx, cy, cx + side, cy + side))

    variants = [
        rgb,
        center_crop,
        ImageOps.autocontrast(center_crop),
    ]

    preds_stack = []
    for variant in variants:
        arr = np.array(variant.resize((224, 224)), dtype=np.float32) / 255.0
        arr = np.expand_dims(arr[:, :, :3], axis=0)
        preds_stack.append(model.predict(arr, verbose=0)[0])

    avg_preds = np.mean(np.stack(preds_stack, axis=0), axis=0)
    idx = int(np.argmax(avg_preds))
    return CLASS_NAMES[idx], float(avg_preds[idx]) * 100.0, avg_preds


def build_disease_analysis(classe: str, confiance: float):
    if classe not in DISEASE_ANALYSIS:
        return None

    analysis = DISEASE_ANALYSIS[classe]
    severity = "Elevee" if confiance >= 80 else "Moderee" if confiance >= 60 else "Faible"
    return {
        "titre": analysis["titre"],
        "symptomes": analysis["symptomes"],
        "cause": analysis["cause"],
        "conseil": analysis["conseil"],
        "severity": severity,
    }


def detect_local_fruit_boxes(img_pil: Image.Image, max_boxes=LOCAL_DETECTION_MAX_BOXES):
    img_rgb = np.array(img_pil.convert("RGB"))
    h0, w0 = img_rgb.shape[:2]
    scale = 1.0

    max_side = max(h0, w0)
    if max_side > LOCAL_MAX_DETECT_SIDE:
        scale = LOCAL_MAX_DETECT_SIDE / float(max_side)
        new_w = max(1, int(w0 * scale))
        new_h = max(1, int(h0 * scale))
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Favor typical fruit colors (red/orange/yellow/green) to reduce false boxes on humans/background.
    red1 = cv2.inRange(hsv, (0, 55, 25), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 55, 25), (179, 255, 255))
    yellow = cv2.inRange(hsv, (15, 55, 30), (40, 255, 255))
    green = cv2.inRange(hsv, (35, 45, 25), (95, 255, 255))
    sat_mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(yellow, green))
    sat_mask = cv2.GaussianBlur(sat_mask, (7, 7), 0)

    kernel = np.ones((5, 5), np.uint8)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = sat_mask.shape
    img_area = float(h * w)
    min_area = LOCAL_DETECTION_MIN_AREA_RATIO * img_area

    detections = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh > 0.95 * img_area:
            continue

        # Reject extreme shapes often produced by people/background instead of fruit blobs.
        aspect = bw / float(max(1, bh))
        fill_ratio = area / float(max(1, bw * bh))
        if aspect < 0.25 or aspect > 4.0:
            continue
        if fill_ratio < 0.22:
            continue

        # Map back to original frame size if detection was done on a resized frame.
        if scale != 1.0:
            x = x / scale
            y = y / scale
            bw = bw / scale
            bh = bh / scale

        detections.append(
            {
                "x": float(x + bw / 2),
                "y": float(y + bh / 2),
                "width": float(bw),
                "height": float(bh),
                "label": "fruit",
                "confidence": 1.0,
            }
        )

        if len(detections) >= max_boxes:
            break

    return detections


def _expand_box(left, top, right, bottom, width, height, margin_ratio=LOCAL_CROP_MARGIN_RATIO):
    bw = right - left
    bh = bottom - top
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    left = max(0, left - mx)
    top = max(0, top - my)
    right = min(width, right + mx)
    bottom = min(height, bottom + my)
    return left, top, right, bottom


def _draw_detections(img_pil: Image.Image, detections, color="#2f80ff"):
    annotated = img_pil.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    normalized = []
    for det in detections:
        x_center = det["x"]
        y_center = det["y"]
        box_w = det["width"]
        box_h = det["height"]

        left = max(0, x_center - box_w / 2)
        top = max(0, y_center - box_h / 2)
        right = min(width, x_center + box_w / 2)
        bottom = min(height, y_center + box_h / 2)

        draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)
        draw.text((left + 4, max(0, top - 18)), f"{det['label']}", fill=color)

        normalized.append(
            {
                **det,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            }
        )

    return annotated, normalized


def classify_detected_regions(img_pil: Image.Image, detections, robust_prediction=True):
    results = []
    img_w, img_h = img_pil.size
    for det in detections:
        left = int(det["left"])
        top = int(det["top"])
        right = int(det["right"])
        bottom = int(det["bottom"])

        left, top, right, bottom = _expand_box(left, top, right, bottom, img_w, img_h)

        crop = img_pil.crop((left, top, right, bottom))
        if robust_prediction:
            classe, confiance, preds = predict_robust(crop, model)
        else:
            classe, confiance, preds = predict(crop, model)
        info = CLASS_INFO[classe]
        comestible = "rotten" not in classe

        # Ignore weak predictions to avoid treating humans/background as fruit detections.
        if confiance < LOCAL_MODEL_MIN_CONFIDENCE:
            continue

        sorted_idx = np.argsort(preds)[::-1]
        top1 = float(preds[sorted_idx[0]]) * 100.0
        top2 = float(preds[sorted_idx[1]]) * 100.0
        margin = top1 - top2

        results.append(
            {
                "api_label": det["label"],
                "classe": classe,
                "nom": info["nom"],
                "confiance": confiance,
                "margin": margin,
                "comestible": comestible,
                "analysis": build_disease_analysis(classe, confiance),
                "box_area": (right - left) * (bottom - top),
            }
        )

    # Prefer bigger valid fruit boxes first, then confidence.
    results.sort(key=lambda item: (item["box_area"], item["confiance"]), reverse=True)
    return results


def render_local_detections(img_pil: Image.Image, max_boxes=LOCAL_DETECTION_MAX_BOXES):
    detections = detect_local_fruit_boxes(img_pil, max_boxes=max_boxes)
    if not detections:
        return None, []

    annotated, normalized = _draw_detections(img_pil, detections, color="#2f80ff")

    return annotated, normalized


def render_model_decisions_on_detections(img_pil: Image.Image, detections):
    if not detections:
        st.info("Aucune detection API exploitable pour cette image.")
        return

    st.markdown("#### 🧠 Decisions du modele local sur les zones detectees")
    max_regions = min(len(detections), 8)

    for idx, det in enumerate(detections[:max_regions], start=1):
        left = int(det["left"])
        top = int(det["top"])
        right = int(det["right"])
        bottom = int(det["bottom"])

        crop = img_pil.crop((left, top, right, bottom))
        classe, confiance, _ = predict(crop, model)
        is_edible = "rotten" not in classe
        verdict = "COMESTIBLE" if is_edible else "NON COMESTIBLE"

        st.markdown(
            f"{idx}. Detection API: **{det['label']}** | Modele local: **{CLASS_INFO[classe]['nom']}** | Verdict: **{verdict}** ({confiance:.1f}%)"
        )

    if len(detections) > max_regions:
        st.caption(f"{len(detections) - max_regions} zones supplementaires non affichees pour garder l'interface legere.")


def render_local_disease_analysis(classe: str, confiance: float):
    disease_analysis = build_disease_analysis(classe, confiance)
    if disease_analysis is None:
        return

    st.markdown(
        f"""
    <div class="result-card non-comestible" style="margin-top:1rem; text-align:left;">
        <div class="verdict-text non-comestible" style="font-size:1.35rem;">🩺 {disease_analysis['titre']}</div>
        <div style="margin-top:0.8rem; color:#f0ede8; font-weight:700;">
            Niveau de suspicion : {disease_analysis['severity']}
        </div>
        <div style="margin-top:0.8rem; color:#c7c2bc;">
            <strong>Symptomes possibles :</strong><br>
            - {disease_analysis['symptomes'][0]}<br>
            - {disease_analysis['symptomes'][1]}<br>
            - {disease_analysis['symptomes'][2]}
        </div>
        <div style="margin-top:0.8rem; color:#c7c2bc;">
            <strong>Cause probable :</strong> {disease_analysis['cause']}
        </div>
        <div style="margin-top:0.8rem; color:#c7c2bc;">
            <strong>Action conseillee :</strong> {disease_analysis['conseil']}
        </div>
        <div style="margin-top:0.8rem; color:#888; font-size:0.8rem;">
            Analyse indicative basee sur la classe predite par le modele actuel, pas un diagnostic medical.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_result_card(classe: str, confiance: float):
    info = CLASS_INFO[classe]
    comestible = "rotten" not in classe
    card_cls = "comestible" if comestible else "non-comestible"
    verdict = "✅ COMESTIBLE" if comestible else "❌ NON COMESTIBLE"

    st.markdown(
        f"""
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
    """,
        unsafe_allow_html=True,
    )

    render_local_disease_analysis(classe, confiance)


class FruitVideoProcessor(VideoProcessorBase):
    def __init__(
        self,
        frame_skip=12,
        infer_interval=1.0,
        detection_interval=0.9,
        max_boxes=LOCAL_DETECTION_MAX_BOXES,
        robust_prediction=True,
    ):
        self.latest_classe = None
        self.latest_confiance = 0.0
        self.latest_preds = None
        self.latest_ts = 0.0
        self.latest_detection_ts = 0.0
        self.next_infer_ts = 0.0
        self.frame_count = 0
        self.frame_skip = max(1, int(frame_skip))
        self.infer_interval = max(0.2, float(infer_interval))
        self.detection_interval = max(0.2, float(detection_interval))
        self.max_boxes = max(1, int(max_boxes))
        self.robust_prediction = bool(robust_prediction)
        self.latest_detections = []
        self.latest_regions = []

    def recv(self, frame):
        self.frame_count += 1
        image = frame.to_ndarray(format="rgb24")
        now = time.time()

        # Time-gated inference prevents spikes and keeps camera stream smooth.
        should_infer = self.latest_ts == 0.0 or now >= self.next_infer_ts

        if should_infer:
            pil_image = Image.fromarray(image)

            detection_due = (now - self.latest_detection_ts) >= self.detection_interval
            if detection_due:
                _, detections = render_local_detections(pil_image, max_boxes=self.max_boxes)
                self.latest_detections = detections
                self.latest_detection_ts = now

                if self.latest_detections:
                    _, normalized = _draw_detections(pil_image, self.latest_detections, color="#2f80ff")
                    self.latest_regions = classify_detected_regions(
                        pil_image,
                        normalized,
                        robust_prediction=self.robust_prediction,
                    )
                else:
                    self.latest_regions = []

            if self.latest_regions:
                best_region = self.latest_regions[0]
                self.latest_classe = best_region["classe"]
                self.latest_confiance = best_region["confiance"]
                self.latest_preds = None
            else:
                self.latest_classe = None
                self.latest_confiance = 0.0
                self.latest_preds = None
            self.latest_ts = now
            self.next_infer_ts = now + self.infer_interval

        # Always draw over the latest camera frame to avoid a frozen/stuttering preview.
        output = Image.fromarray(image)
        if self.latest_detections and (now - self.latest_detection_ts) <= (self.detection_interval * 2.0):
            output, _ = _draw_detections(output, self.latest_detections, color="#2f80ff")
        output_arr = np.array(output)
        output_bgr = cv2.cvtColor(output_arr, cv2.COLOR_RGB2BGR)

        if self.latest_regions:
            best = self.latest_regions[0]
            verdict = "COMESTIBLE" if best["comestible"] else "NON COMESTIBLE"
            text = f"{best['nom']} | {verdict} | {best['confiance']:.1f}%"
            color = (46, 204, 113) if best["comestible"] else (231, 76, 60)
            cv2.rectangle(output_bgr, (10, 10), (min(output_bgr.shape[1] - 10, 620), 48), (15, 20, 15), -1)
            cv2.putText(output_bgr, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(output_bgr, (10, 10), (420, 48), (20, 20, 20), -1)
            cv2.putText(
                output_bgr,
                "Aucun fruit detecte",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), format="rgb24")


model_data = load_fruit_model()
if model_data is None or model_data[0] is None:
    st.error("Modele introuvable. Place best_model_phase1.keras a la racine du projet.")
    st.write("Fichiers disponibles:", os.listdir("."))
    st.stop()

model, active_model_path = model_data


st.markdown(
    """
<div class="main-header">
    <p class="main-title">FreshScan</p>
    <p class="main-subtitle">Detecteur de fraicheur - IA MobileNetV2</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="info-grid">
    <div class="info-card">
        <div class="info-icon">🧠</div>
        <div class="info-title">Modele</div>
        <div class="info-value">MobileNetV2</div>
    </div>
    <div class="info-card">
        <div class="info-icon">🎯</div>
        <div class="info-title">Precision</div>
        <div class="info-value">99.86%</div>
    </div>
    <div class="info-card">
        <div class="info-icon">🍓</div>
        <div class="info-title">Classes</div>
        <div class="info-value">6 types</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption(f"Model active: {active_model_path}")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

mode_options = ["Photo"]
if HAS_WEBRTC:
    mode_options.append("Temps reel (webcam)")

mode = st.radio(
    "Choisir le mode d'analyse",
    mode_options,
    horizontal=True,
)

if mode == "Temps reel (webcam)":
    selected_perf = {
        "frame_skip": 12,
        "infer_interval": 1.0,
        "detection_interval": 0.9,
        "frame_rate": 12,
        "width": 640,
        "height": 360,
        "max_boxes": 2,
        "robust_prediction": True,
    }
    st.caption("Mode webcam leger actif (16:9): cadrage proche de la camera Windows, avec traitement allege.")

    st.markdown(
        """
    <div style="margin:0.5rem 0 1rem 0; color:#888; font-size:0.9rem;">
        Active ta camera pour analyser le fruit en continu avec le modele actuel.
    </div>
    """,
        unsafe_allow_html=True,
    )

    if HAS_WEBRTC:
        ctx = webrtc_streamer(
            key="freshscan-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: FruitVideoProcessor(
                frame_skip=selected_perf["frame_skip"],
                infer_interval=selected_perf["infer_interval"],
                detection_interval=selected_perf["detection_interval"],
                max_boxes=selected_perf["max_boxes"],
                robust_prediction=selected_perf["robust_prediction"],
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": selected_perf["width"]},
                    "height": {"ideal": selected_perf["height"]},
                    "frameRate": {"ideal": selected_perf["frame_rate"], "max": selected_perf["frame_rate"]},
                    "aspectRatio": {"ideal": 1.7777777778},
                    "resizeMode": "none",
                    "facingMode": "user",
                },
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.video_processor is not None:
            st.markdown("#### Resultat en direct")
            st.caption("Verdict affiche sur la video. Interface simplifiee pour eviter les bugs.")
            if ctx.video_processor.latest_regions:
                best_region = ctx.video_processor.latest_regions[0]
                st.markdown(
                    f"**Detection locale:** {best_region['api_label']} | **Modele local:** {CLASS_INFO[best_region['classe']]['nom']} | **Confiance:** {best_region['confiance']:.1f}%"
                )
            else:
                st.warning("Aucun fruit detecte pour le moment. Cadrez un fruit au centre pour voir le rectangle bleu.")

            st.caption(
                f"Inference toutes ~{selected_perf['infer_interval']}s | "
                f"Video {selected_perf['width']}x{selected_perf['height']}"
            )
    else:
        st.info("Le mode webcam n'est pas disponible dans cette installation allégée. Utilise le mode Photo.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Le flux peut mettre quelques secondes a se stabiliser selon le navigateur.")

    st.stop()

uploaded = st.file_uploader(
    "Depose une photo de fruit ici",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible",
)

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.image(img_pil, use_container_width=True, caption="Photo analysee")

    with col2:
        with st.spinner("Analyse en cours..."):
            time.sleep(0.5)
            classe, confiance, all_preds = predict(img_pil, model)
        render_result_card(classe, confiance)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📊 Probabilites par classe")

    fig = go.Figure(
        go.Bar(
            x=[p * 100 for p in all_preds],
            y=CLASS_NAMES,
            orientation="h",
            marker=dict(
                color=[BAR_COLORS[c] for c in CLASS_NAMES],
                opacity=0.85,
                line=dict(width=0),
            ),
            text=[f"{p * 100:.1f}%" for p in all_preds],
            textposition="outside",
            textfont=dict(color="#888", size=12),
        )
    )

    fig.update_layout(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#13131a",
        font=dict(family="DM Sans", color="#888"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#1e1e2e",
            range=[0, 110],
            ticksuffix="%",
            color="#555",
        ),
        yaxis=dict(color="#aaa", tickfont=dict(size=13)),
        margin=dict(l=10, r=40, t=20, b=20),
        height=280,
        showlegend=False,
        bargap=0.35,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Analyser une autre photo"):
        st.rerun()
else:
    st.markdown(
        """
    <div style="text-align:center; padding: 3rem 0; color: #333;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">🍎🍌🍊</div>
        <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; color: #444;">
            Charge une photo pour commencer l'analyse
        </div>
        <div style="font-size: 0.85rem; color: #333; margin-top: 0.5rem;">
            Formats acceptes : JPG, PNG, WEBP
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="footer">
    FRESHSCAN · MOBILENETV2 · TRANSFER LEARNING · 99.86% ACCURACY
</div>
""",
    unsafe_allow_html=True,
)