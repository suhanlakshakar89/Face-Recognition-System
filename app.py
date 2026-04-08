import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

st.set_page_config(page_title="Face Attendance", page_icon="🎓")
st.title("🎓 Face Recognition Attendance System")

# ── Cell 2+3: Load dataset ────────────────────────────────────
DATASET_PATH = "Images_dataset"
ATTENDANCE_FILE = "attendance.csv"

@st.cache_resource(show_spinner="Loading face encodings...")
def load_encodings():
    images, classNames = [], []
    for person in os.listdir(DATASET_PATH):
        person_path = f"{DATASET_PATH}/{person}"
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img = cv2.imread(f"{person_path}/{img_name}")
            if img is not None:
                images.append(img)
                classNames.append(person)

    # ── Cell 5: findEncodings ─────────────────────────────────
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if len(encodes) > 0:
            encodeList.append(encodes[0])

    return encodeList, classNames

encodeListKnown, classNames = load_encodings()
st.sidebar.success(f"✅ Encoding Complete\nLoaded: {set(classNames)}")

# ── Cell 4: markAttendance ────────────────────────────────────
def markAttendance(name):
    with open(ATTENDANCE_FILE, 'a+') as f:
        f.seek(0)
        data = f.readlines()
        names = [line.split(',')[0] for line in data]
        if name not in names:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            date = now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{date},{time}')

# ── Cell 7: Upload & Detect ───────────────────────────────────
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.write("📁 Image uploaded successfully")

    if st.button("🔍 Detect Face", type="primary"):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgRGB, model='hog')
        encodesCurFrame = face_recognition.face_encodings(imgRGB, facesCurFrame)

        if len(encodesCurFrame) == 0:
            st.error("❌ No face detected in the image")
        else:
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                if len(faceDis) == 0:
                    st.error("❌ No known faces to compare")
                    continue

                matchIndex = np.argmin(faceDis)
                st.write(f"🔍 Distance Score: `{faceDis[matchIndex]:.4f}` (threshold = 0.5)")

                if faceDis[matchIndex] < 0.5:
                    name = classNames[matchIndex].upper()
                    now = datetime.now()
                    time_str = now.strftime('%H:%M:%S')
                    date_str = now.strftime('%d/%m/%Y')

                    st.success(f"✅ Detected Person: **{name}**")
                    st.write(f"🕒 Time: {time_str}")
                    st.write(f"📅 Date: {date_str}")
                    st.info("📌 Attendance Marked")

                    markAttendance(name)
                    color = (0, 255, 0)
                else:
                    name = "UNKNOWN"
                    st.error("❌ Imposter / Not Recognized")
                    color = (0, 0, 255)

                # Draw rectangle on image (same as notebook)
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="Face Recognition Result",
                     use_container_width=True)
