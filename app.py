# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# import os
# from datetime import datetime

# # Page Config
# st.set_page_config(page_title="Face Attendance System", layout="centered")
# st.title("📸 Face Recognition Attendance")

# # 1. Load & Encode Known Faces (Cached to prevent timeout)
# @st.cache_data
# def load_known_data(path='Images_dataset'):
#     known_encodings = []
#     known_names = []
    
#     if not os.path.exists(path):
#         return [], []

#     for person in os.listdir(path):
#         person_path = os.path.join(path, person)
#         if os.path.isdir(person_path):
#             for img_name in os.listdir(person_path):
#                 img = cv2.imread(os.path.join(person_path, img_name))
#                 if img is not None:
#                     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                     encodes = face_recognition.face_encodings(img_rgb)
#                     if encodes:
#                         known_encodings.append(encodes[0])
#                         known_names.append(person.upper())
#     return known_encodings, known_names

# # 2. Attendance Logic
# def markAttendance(name):
#     file_path = 'attendance.csv'
#     # Create file with header if not exists
#     if not os.path.exists(file_path):
#         with open(file_path, 'w') as f:
#             f.write('Name,Date,Time')

#     with open(file_path, 'r+') as f:
#         data = f.readlines()
#         name_list = [line.split(',')[0] for line in data]
#         if name not in name_list:
#             now = datetime.now()
#             f.write(f'\n{name},{now.strftime("%d/%m/%Y")},{now.strftime("%H:%M:%S")}')
#             return True
#     return False

# # Initialize Data
# encodeListKnown, classNames = load_known_data()

# # 3. UI - Image Upload
# uploaded_file = st.file_uploader("📤 Upload Image for Recognition", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert uploaded file to OpenCV image
#     file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
#     img = cv2.imdecode(file_bytes, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     with st.spinner('🔍 Scanning faces...'):
#         face_locs = face_recognition.face_locations(img_rgb)
#         face_encodes = face_recognition.face_encodings(img_rgb, face_locs)

#     if not face_encodes:
#         st.error("❌ No face detected in the image.")
#     else:
#         for encodeFace, faceLoc in zip(face_encodes, face_locs):
#             faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            
#             if len(faceDis) > 0:
#                 matchIndex = np.argmin(faceDis)
                
#                 # Threshold 0.5
#                 if faceDis[matchIndex] < 0.5:
#                     name = classNames[matchIndex]
#                     is_new = markAttendance(name)
#                     color = (0, 255, 0) # Green
#                     st.success(f"✅ Detected: {name}")
#                     if is_new: st.info(f"📌 Attendance marked for {name}")
#                 else:
#                     name = "UNKNOWN"
#                     color = (255, 0, 0) # Red
#                     st.warning("⚠️ Imposter / Not Recognized")

#                 # Draw Box on Image
#                 y1, x2, y2, x1 = faceLoc
#                 cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(img_rgb, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#         st.image(img_rgb, caption='Processed Result', use_column_width=True) 


import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime

st.set_page_config(page_title="Face Attendance System", layout="centered")
st.title("📸 Face Recognition Attendance")

ENCODING_FILE = "encodings.pkl"
DATASET_PATH = "Images_dataset"

# -----------------------------
# LOAD OR CREATE ENCODINGS
# -----------------------------
@st.cache_data
def load_or_create_encodings():
    if os.path.exists(ENCODING_FILE):
        with open(ENCODING_FILE, "rb") as f:
            data = pickle.load(f)
            return data["encodings"], data["names"]

    known_encodings = []
    known_names = []

    if not os.path.exists(DATASET_PATH):
        st.error("❌ Dataset folder not found!")
        return [], []

    with st.spinner("⚙️ Encoding faces (first time only)..."):
        for person in os.listdir(DATASET_PATH):
            person_path = os.path.join(DATASET_PATH, person)

            if os.path.isdir(person_path):
                for img_name in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_name)
                    img = cv2.imread(img_path)

                    if img is None:
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encodes = face_recognition.face_encodings(img_rgb)

                    if encodes:
                        known_encodings.append(encodes[0])
                        known_names.append(person.upper())

    # Save encodings
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    return known_encodings, known_names


# -----------------------------
# ATTENDANCE
# -----------------------------
def markAttendance(name):
    file_path = "attendance.csv"

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("Name,Date,Time")

    with open(file_path, "r+") as f:
        data = f.readlines()
        name_list = [line.split(",")[0] for line in data]

        if name not in name_list:
            now = datetime.now()
            f.write(f"\n{name},{now.strftime('%d/%m/%Y')},{now.strftime('%H:%M:%S')}")
            return True
    return False


# -----------------------------
# LOAD DATA
# -----------------------------
encodeListKnown, classNames = load_or_create_encodings()

if len(encodeListKnown) == 0:
    st.warning("⚠️ No faces found in dataset. Check Images_dataset folder.")


# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with st.spinner("🔍 Detecting faces..."):
        face_locs = face_recognition.face_locations(img_rgb)
        face_encodes = face_recognition.face_encodings(img_rgb, face_locs)

    if not face_encodes:
        st.error("❌ No face detected!")
    else:
        for encodeFace, faceLoc in zip(face_encodes, face_locs):
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            if len(faceDis) > 0:
                matchIndex = np.argmin(faceDis)

                # 🔥 Improved threshold
                if faceDis[matchIndex] < 0.45:
                    name = classNames[matchIndex]
                    is_new = markAttendance(name)
                    color = (0, 255, 0)

                    st.success(f"✅ {name}")
                    if is_new:
                        st.info("📌 Attendance Marked")

                else:
                    name = "UNKNOWN"
                    color = (255, 0, 0)
                    st.warning("❌ Not Recognized")

                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_rgb, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        st.image(img_rgb, caption="Result", use_column_width=True)
