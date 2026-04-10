import streamlit as st
import face_recognition
import numpy as np
import cv2
import pandas as pd
import os
from datetime import datetime
from PIL import Image

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="🧑‍💼",
    layout="centered"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
h1, h2, h3                  { font-family: 'Space Mono', monospace; letter-spacing: -0.5px; }

.stButton > button {
    background: #111; color: #fff;
    border: none; border-radius: 8px;
    padding: 0.55rem 2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem; letter-spacing: 0.5px;
    transition: background 0.2s;
    width: 100%;
}
.stButton > button:hover { background: #333; }

.card {
    padding: 1.2rem 1.4rem;
    border-radius: 10px;
    margin: 0.8rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.88rem;
    line-height: 1.9;
}
.card-success { background: #f0fdf4; border-left: 5px solid #22c55e; }
.card-danger  { background: #fef2f2; border-left: 5px solid #ef4444; }
.card-info    { background: #f0f9ff; border-left: 5px solid #3b82f6; }

.step-header {
    background: #111; color: #fff;
    padding: 0.5rem 1rem; border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem; margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ATTENDANCE_FILE = "attendance.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────
def mark_attendance(name: str) -> pd.DataFrame:
    now = datetime.now()
    row = {
        "Name": name,
        "Date": now.strftime("%d/%m/%Y"),
        "Time": now.strftime("%H:%M:%S")
    }
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    if name not in df["Name"].values:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
    return df


def encode_uploaded_images(files) -> tuple:
    encodings, names = [], []
    for f in files:
        img = np.array(Image.open(f).convert("RGB"))
        encs = face_recognition.face_encodings(img)
        name = os.path.splitext(f.name)[0].replace("_", " ").title()
        if encs:
            encodings.append(encs[0])
            names.append(name)
    return encodings, names


def load_attendance() -> pd.DataFrame:
    try:
        return pd.read_csv(ATTENDANCE_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Name", "Date", "Time"])

# ── Session State ──────────────────────────────────────────────────────────────
if "known_encodings" not in st.session_state:
    st.session_state.known_encodings = []
if "known_names" not in st.session_state:
    st.session_state.known_names = []
if "registered_names" not in st.session_state:
    st.session_state.registered_names = []

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧑‍💼 Face Recognition Attendance")
st.caption("Upload known faces → Upload test photo → Get result with date & time")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — REGISTER KNOWN FACES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-header">STEP 1 — Register Known Faces</div>', unsafe_allow_html=True)
st.caption("Upload one or more photos. **Filename = person's name** (e.g. `suhan.jpg`, `john_doe.jpg`)")

known_files = st.file_uploader(
    "Upload known face images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="known_upload"
)

if st.button("📁  Register Faces"):
    if not known_files:
        st.warning("Please upload at least one image.")
    else:
        with st.spinner("Encoding faces…"):
            encs, names = encode_uploaded_images(known_files)

        if encs:
            st.session_state.known_encodings  = encs
            st.session_state.known_names      = names
            st.session_state.registered_names = list(set(names))
            st.markdown(
                f'<div class="card card-info">✅ Registered successfully:<br><b>{", ".join(set(names))}</b></div>',
                unsafe_allow_html=True
            )
        else:
            st.error("No faces detected in the uploaded images. Try clearer/closer photos.")

# Show currently registered names
if st.session_state.registered_names:
    st.success(f"Currently registered: **{', '.join(st.session_state.registered_names)}**")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — DETECT & MARK ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-header">STEP 2 — Upload Photo to Detect</div>', unsafe_allow_html=True)
st.caption("Upload a photo of the person you want to identify.")

test_file = st.file_uploader(
    "Upload photo for detection",
    type=["jpg", "jpeg", "png"],
    key="test_upload"
)

if st.button("🔍  Detect Face & Mark Attendance"):
    if test_file is None:
        st.warning("Please upload a photo to detect.")
    elif not st.session_state.known_encodings:
        st.warning("No known faces registered yet. Complete Step 1 first.")
    else:
        with st.spinner("Detecting face…"):
            img_pil = Image.open(test_file).convert("RGB")
            img_np  = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            face_locs = face_recognition.face_locations(img_np, model="hog")
            face_encs = face_recognition.face_encodings(img_np, face_locs)

        if not face_encs:
            st.markdown(
                '<div class="card card-danger">❌ No face detected in this image.<br>Try a clearer, well-lit photo.</div>',
                unsafe_allow_html=True
            )
        else:
            for enc, loc in zip(face_encs, face_locs):
                distances = face_recognition.face_distance(
                    st.session_state.known_encodings, enc
                )
                best_idx  = int(np.argmin(distances))
                best_dist = float(distances[best_idx])

                y1, x2, y2, x1 = loc

                now       = datetime.now()
                time_str  = now.strftime("%H:%M:%S")
                date_str  = now.strftime("%d/%m/%Y")

                if best_dist < 0.50:
                    # ── MATCH ──────────────────────────────────────────────
                    name  = st.session_state.known_names[best_idx].upper()
                    color = (34, 197, 94)      # green

                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_bgr, name, (x1, y1 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    mark_attendance(name)

                    st.markdown(
                        f'<div class="card card-success">'
                        f'✅ &nbsp;<b>Detected Person:</b> {name}<br>'
                        f'🔍 &nbsp;<b>Distance Score:</b> {best_dist:.4f} &nbsp;(lower = better match)<br>'
                        f'🕒 &nbsp;<b>Time:</b> {time_str}<br>'
                        f'📅 &nbsp;<b>Date:</b> {date_str}<br>'
                        f'📌 &nbsp;<b>Attendance Marked Successfully</b>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                else:
                    # ── NO MATCH ───────────────────────────────────────────
                    name  = "UNKNOWN"
                    color = (239, 68, 68)      # red

                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_bgr, "UNKNOWN", (x1, y1 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    st.markdown(
                        f'<div class="card card-danger">'
                        f'❌ &nbsp;<b>Imposter / Not Recognized</b><br>'
                        f'🔍 &nbsp;<b>Distance Score:</b> {best_dist:.4f} &nbsp;(threshold = 0.50)<br>'
                        f'🕒 &nbsp;<b>Time:</b> {time_str}<br>'
                        f'📅 &nbsp;<b>Date:</b> {date_str}<br>'
                        f'🚫 &nbsp;<b>Attendance NOT Marked</b>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # ── Show annotated image ───────────────────────────────────────
            result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="Detection Result", use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — ATTENDANCE LOG
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-header">STEP 3 — Attendance Log</div>', unsafe_allow_html=True)

df = load_attendance()

if df.empty:
    st.info("No attendance records yet. Records will appear here after detection.")
else:
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Attendance CSV",
        data=csv_bytes,
        file_name="attendance.csv",
        mime="text/csv"
    )
