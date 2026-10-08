import time
import threading
import urllib.request
from collections import deque
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="SignLearn Healthcare", page_icon="🤟", layout="wide")

# ---------------------------------------------------------
# CONFIG  (edit to match your training setup)
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "stgcn_improved_model.pt"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
SEQ_LEN = 30
THRESHOLD = 0.50
FLIP_INPUT = False   # set True only if your training data was mirrored (flipped) first
# ORDER MUST MATCH THE CLASS INDICES USED DURING TRAINING
LABELS = ["HELLO", "HELP", "WATER", "FOOD", "PAIN",
          "DOCTOR", "MEDICINE", "HOSPITAL", "YES", "NO"]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# ---------------------------------------------------------
# MODEL HELPERS
# ---------------------------------------------------------
def ensure_hand_model():
    if not HAND_MODEL_PATH.exists():
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    import torch
    import sys

    sys.path.insert(0, str(MODEL_DIR))
    from stgcn_model import STGCNModel

    model = STGCNModel(
        input_channels=3,
        hidden_channels=64,
        num_classes=10,
        dropout=0.3,
    )

    checkpoint = torch.load(
        str(MODEL_PATH),
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def normalize(pts: np.ndarray) -> np.ndarray:
    """pts: (21, 3). Wrist-relative, scale-invariant.
    CHANGE THIS to match the preprocessing used in training."""
    pts = pts - pts[0]
    scale = np.linalg.norm(pts, axis=1).max()
    return pts / scale if scale > 1e-6 else pts


def predict(model, seq: np.ndarray):
    """seq: (30, 21, 3) -> (label, confidence).
    Assumes input shape (N, C, T, V, M) = (1, 3, 30, 21, 1)."""
    import torch
    x = torch.from_numpy(seq).float().unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    i = int(probs.argmax())
    return LABELS[i], float(probs[i])


# ---------------------------------------------------------
# VIDEO PROCESSOR (background thread)
# ---------------------------------------------------------
class SignProcessor(VideoProcessorBase):
    def __init__(self, model):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.model = model
        self.buffer = deque(maxlen=SEQ_LEN)
        self.missing = 0
        self.frames = 0
        self.lock = threading.Lock()
        self.state = {"hand": False, "buffered": 0, "word": None, "conf": 0.0, "error": None}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if FLIP_INPUT:
            img = cv2.flip(img, 1)
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = self.detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                self.frames,
            )
        self.frames += 1
        h, w = img.shape[:2]
        word, conf, hand = None, 0.0, False

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            hand = True
            self.missing = 0
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            self.buffer.append(normalize(pts))
            px = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, px[a], px[b], (0, 255, 0), 2)
            for c in px:
                cv2.circle(img, c, 4, (0, 0, 255), -1)
        else:
            self.missing += 1
            if self.missing > 10:        # hand left the frame: start a fresh sequence
                self.buffer.clear()

        err = None
        if self.model is not None and len(self.buffer) == SEQ_LEN and self.frames % 5 == 0:
            try:
                word, conf = predict(self.model, np.stack(self.buffer))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

        with self.lock:
            self.state["hand"] = hand
            self.state["buffered"] = len(self.buffer)
            if word is not None:
                self.state["word"], self.state["conf"] = word, conf
                self.state["error"] = None
            if err:
                self.state["error"] = err

        # mirror only for display so it feels like a mirror
        return av.VideoFrame.from_ndarray(cv2.flip(img, 1), format="bgr24")


# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------
st.markdown(
    """<style>
.stApp{background:#F5F7FB}
.block-container{padding-top:2rem;max-width:1400px}
.header{background:linear-gradient(135deg,#2563EB,#1D4ED8);padding:32px;border-radius:20px;color:white;margin-bottom:25px}
.header h1{margin:0;padding:0;color:white;font-size:36px;font-weight:800}
.header p{color:white;margin:8px 0 0}
.card{background:white;color:#111827;border-radius:18px;padding:24px;margin-bottom:20px;border:1px solid #E5E7EB;box-shadow:0 4px 18px rgba(0,0,0,.06)}
.card-title{font-size:20px;font-weight:700;color:#111827;margin-bottom:6px}
.sub{font-size:14px;color:#6B7280}
.recog{background:#F8FAFC;border-radius:16px;padding:25px;text-align:center;border:1px solid #E5E7EB}
.label{font-size:14px;color:#6B7280;text-transform:uppercase;letter-spacing:1px;font-weight:700}
.word{font-size:42px;font-weight:800;margin:15px 0}
.pill{display:inline-block;background:#EFF6FF;color:#1D4ED8;border:1px solid #DBEAFE;border-radius:30px;padding:8px 14px;margin:4px;font-size:13px;font-weight:700}
</style>""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="header"><h1>🤟 SignLearn Healthcare</h1>'
    "<p>Word-Level Sign Language Recognition for Healthcare Communication</p></div>",
    unsafe_allow_html=True,
)

# Download the hand-tracking model once (needs internet the first time)
try:
    with st.spinner("Preparing hand tracking model..."):
        ensure_hand_model()
except Exception as e:
    st.error(
        f"Could not download the hand tracking model: {e}\n\n"
        f"Download it manually from:\n{HAND_MODEL_URL}\n"
        f"and save it as `{HAND_MODEL_PATH}` next to this script."
    )
    st.stop()

model = load_model()
if model is None:
    st.warning(
        f"`{MODEL_PATH}` not found. The camera and hand tracking will work, "
        "but no words can be recognized until you place your trained model there."
    )

left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown(
        '<div class="card-title">📷 Live Camera</div>'
        '<div class="sub">Click START, allow camera access, and keep one hand clearly in view.</div>',
        unsafe_allow_html=True,
    )
    ctx = webrtc_streamer(
        key="signlearn",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: SignProcessor(model),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    status_ph = st.empty()
    result_ph = st.empty()
    debug_ph = st.empty()

with right:
    history_ph = st.empty()
    pills = "".join(f'<span class="pill">{w}</span>' for w in LABELS)
    st.markdown(
        f'<div class="card"><div class="card-title">🏥 Supported Healthcare Signs</div>{pills}</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# LIVE UI UPDATE LOOP
# ---------------------------------------------------------
history = st.session_state.setdefault("history", [])


def render_history():
    items = "".join(
        f'<div class="sub">{w} · {c:.0%}</div>' for w, c in reversed(history[-8:])
    ) or '<div class="sub">No recognition yet.</div>'
    history_ph.markdown(
        f'<div class="card"><div class="card-title">🕐 Recognition History</div>{items}</div>',
        unsafe_allow_html=True,
    )


def render_result(word, conf):
    shown = word if word and conf >= THRESHOLD else "WAITING..."
    color = "#16A34A" if shown != "WAITING..." else "#9CA3AF"
    result_ph.markdown(
        f'<div class="card"><div class="recog"><div class="label">Current Recognition</div>'
        f'<div class="word" style="color:{color}">{shown}</div>'
        f'<div>Confidence: <strong>{conf:.1%}</strong></div></div></div>',
        unsafe_allow_html=True,
    )


render_history()
render_result(None, 0.0)

last_logged = None
while ctx.state.playing:
    proc = ctx.video_processor
    if proc is not None:
        with proc.lock:
            s = dict(proc.state)
        if s["hand"]:
            status_ph.info(f"✋ Hand detected · collecting frames {s['buffered']}/{SEQ_LEN}")
        else:
            status_ph.warning("No hand detected. Move your hand into the camera view.")
        render_result(s["word"], s["conf"])
        raw = f"{s['word']} ({s['conf']:.0%})" if s["word"] else "none yet"
        debug_ph.caption(
            f"🔧 Debug · model loaded: {'yes' if model is not None else 'NO'} · "
            f"frames buffered: {s['buffered']}/{SEQ_LEN} · "
            f"raw prediction: {raw} · threshold: {THRESHOLD:.0%}"
            + (f" · ⚠️ prediction error: {s['error']}" if s["error"] else "")
        )
        if s["word"] and s["conf"] >= THRESHOLD and s["word"] != last_logged:
            history.append((s["word"], s["conf"]))
            last_logged = s["word"]
            render_history()
        if not s["hand"]:
            last_logged = None
    time.sleep(0.2)