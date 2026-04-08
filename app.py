import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Face Attendance", page_icon="🎓")
st.title("🎓 Face Recognition Attendance System")

# ── Load encodings (same logic as notebook) ──────────────────
DATASET_PATH = "Images_dataset"
ATTENDANCE_FILE = "attendance.csv"

@st.cache_resource(show_spinner="Loading face encodings...")
def load_encodings():
    images, class_names = [], []
    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img = cv2.imread(os.path.join(person_path, img_name))
            if img is not None:
                images.append(img)
                class_names.append(person)

    encode_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img_rgb)
        if enc:
            encode_list.append(enc[0])

    return encode_list, class_names

encode_list_known, class_names = load_encodings()
st.sidebar.success(f"✅ Loaded: {', '.join(set(class_names)) or 'None'}")

# ── Mark attendance (same logic as notebook) ─────────────────
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%d/%m/%Y")
    time_str = now.strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    already = ((df["Name"].str.upper() == name.upper()) & (df["Date"] == date_str)).any()
    if not already:
        new_row = pd.DataFrame([{"Name": name, "Date": date_str, "Time": time_str}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return True
    return False

# ── Upload & Detect ───────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a photo to detect", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("Detecting..."):
        face_locations = face_recognition.face_locations(img_rgb, model="hog")
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

    if not face_encodings:
        st.error("❌ No face detected in the image.")
    else:
        for encode_face, face_loc in zip(face_encodings, face_locations):
            face_dis = face_recognition.face_distance(encode_list_known, encode_face)
            match_index = int(np.argmin(face_dis))
            distance = float(face_dis[match_index])

            st.write(f"🔍 Distance Score: `{distance:.4f}` (threshold = 0.5)")

            if distance < 0.5:
                name = class_names[match_index].upper()
                now = datetime.now()
                st.success(f"✅ Detected: **{name}**")
                st.write(f"📅 {now.strftime('%d/%m/%Y')}  🕒 {now.strftime('%H:%M:%S')}")

                if mark_attendance(name):
                    st.info("📌 Attendance Marked!")
                else:
                    st.warning("⚠️ Already marked today.")

                color = (0, 255, 0)
            else:
                name = "UNKNOWN"
                st.error("❌ Imposter / Not Recognized")
                color = (0, 0, 255)

            y1, x2, y2, x1 = face_loc
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_bgr, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)

# ── Attendance Table ──────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Attendance Log")
if os.path.exists(ATTENDANCE_FILE):
    st.dataframe(pd.read_csv(ATTENDANCE_FILE), use_container_width=True)
else:
    st.info("No records yet.")
