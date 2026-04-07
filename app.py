# import cv2
# import face_recognition
# import numpy as np
# import os
# import streamlit as st
# from datetime import datetime
# import pandas as pd

# # ─── Page config ───────────────────────────────────────────────
# st.set_page_config(
#     page_title="Face Attendance System",
#     page_icon="🎓",
#     layout="centered"
# )

# # ─── Custom CSS ────────────────────────────────────────────────
# st.markdown("""
# <style>
#     .main-title {
#         text-align: center;
#         font-size: 2.2rem;
#         font-weight: 700;
#         margin-bottom: 0.2rem;
#     }
#     .sub-title {
#         text-align: center;
#         color: gray;
#         margin-bottom: 1.5rem;
#         font-size: 1rem;
#     }
#     .result-box {
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 0.5rem 0;
#     }
#     div[data-testid="stTabs"] button {
#         font-size: 1rem;
#         font-weight: 600;
#         padding: 0.5rem 1.5rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ─── Header ────────────────────────────────────────────────────
# st.markdown('<div class="main-title">🎓 Face Attendance System</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub-title">Automatic attendance using face recognition</div>', unsafe_allow_html=True)
# st.divider()

# # ─── Load encodings (cached — only runs once) ──────────────────
# @st.cache_resource
# def load_encodings():
#     path = 'Images_dataset'
#     images, classNames = [], []

#     if not os.path.exists(path):
#         st.error("❌ 'Images_dataset' folder not found!")
#         return [], []

#     for person in os.listdir(path):
#         person_path = f'{path}/{person}'
#         if not os.path.isdir(person_path):
#             continue
#         for img_name in os.listdir(person_path):
#             full_path = f'{person_path}/{img_name}'
#             img = cv2.imread(full_path)
#             if img is not None:
#                 images.append(img)
#                 classNames.append(person)

#     encodeList = []
#     for img in images:
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encodes = face_recognition.face_encodings(img_rgb)
#         if encodes:
#             encodeList.append(encodes[0])

#     return encodeList, classNames

# # ─── Attendance helper ─────────────────────────────────────────
# def markAttendance(name):
#     file = 'attendance.csv'
#     now = datetime.now()
#     time_str = now.strftime('%H:%M:%S')
#     date_str = now.strftime('%d/%m/%Y')

#     existing_names = []
#     if os.path.exists(file) and os.path.getsize(file) > 0:
#         try:
#             df = pd.read_csv(file)
#             existing_names = df['Name'].tolist()
#         except:
#             pass

#     with open(file, 'a') as f:
#         if not os.path.exists(file) or os.path.getsize(file) == 0:
#             f.write("Name,Date,Time\n")
#         if name not in existing_names:
#             f.write(f"{name},{date_str},{time_str}\n")
#             return True
#     return False

# # ─── Face detection core ───────────────────────────────────────
# def detect_faces(img_bytes, encodeListKnown, classNames):
#     file_bytes = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     face_locs = face_recognition.face_locations(img_rgb, model='hog')
#     face_encs = face_recognition.face_encodings(img_rgb, face_locs)

#     if not face_encs:
#         return None, []

#     result_img = img.copy()
#     results = []

#     for encodeFace, faceLoc in zip(face_encs, face_locs):
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

#         if len(faceDis) == 0:
#             name = "UNKNOWN"
#             color = (0, 0, 255)
#             results.append({
#                 "name": name,
#                 "distance": "-",
#                 "status": "❌ Not recognized / Imposter"
#             })
#         else:
#             matchIndex = np.argmin(faceDis)
#             distance = faceDis[matchIndex]

#             if distance < 0.5:
#                 name = classNames[matchIndex].upper()
#                 color = (0, 220, 0)
#                 newly_marked = markAttendance(classNames[matchIndex])
#                 results.append({
#                     "name": name,
#                     "distance": round(distance, 4),
#                     "status": "✅ Marked" if newly_marked else "⚠️ Already marked today"
#                 })
#             else:
#                 name = "UNKNOWN"
#                 color = (0, 0, 255)
#                 results.append({
#                     "name": name,
#                     "distance": round(distance, 4),
#                     "status": "❌ Not recognized / Imposter"
#                 })

#         # Draw rectangle + name
#         y1, x2, y2, x1 = faceLoc
#         cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
#         cv2.rectangle(result_img, (x1, y2 - 30), (x2, y2), color, cv2.FILLED)
#         cv2.putText(result_img, name, (x1 + 6, y2 - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

#     return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), results

# # ─── Show results helper ───────────────────────────────────────
# def show_results(result_img, results):
#     if result_img is None:
#         st.error("❌ No face detected in the image. Try again.")
#         return

#     st.image(result_img, caption="Detection Result", use_column_width=True)
#     st.subheader("🔍 Result")

#     for r in results:
#         dist_text = f"  |  Confidence: `{r['distance']}`" if r['distance'] != '-' else ""
#         if "UNKNOWN" in r["name"]:
#             st.error(f"**{r['name']}** — {r['status']}{dist_text}")
#         elif "Already" in r["status"]:
#             st.warning(f"**{r['name']}** — {r['status']}{dist_text}")
#         else:
#             st.success(f"**{r['name']}** — {r['status']}{dist_text}")

# # ══════════════════════════════════════════════════════════════
# #  LOAD ENCODINGS
# # ══════════════════════════════════════════════════════════════
# with st.spinner("⏳ Loading face encodings from dataset..."):
#     encodeListKnown, classNames = load_encodings()

# if classNames:
#     unique_names = list(set(classNames))
#     cols = st.columns(len(unique_names))
#     for i, name in enumerate(unique_names):
#         with cols[i]:
#             st.metric(label="👤 Person", value=name)
# else:
#     st.warning("No encodings loaded. Check your Images_dataset folder.")

# st.divider()

# # ══════════════════════════════════════════════════════════════
# #  TABS: CAMERA | UPLOAD
# # ══════════════════════════════════════════════════════════════
# tab1, tab2 = st.tabs(["📷  Camera Capture", "📁  Upload Image"])

# # ─────────────────────────────────────────
# # TAB 1 — CAMERA
# # ─────────────────────────────────────────
# with tab1:
#     st.markdown("#### Take a photo to mark attendance")
#     st.caption("Click **Take Photo** below, then press **Detect Face**.")

#     camera_image = st.camera_input(" ")  # built-in Streamlit webcam widget

#     if camera_image is not None:
#         if st.button("🔍 Detect Face", key="cam_detect", use_container_width=True):
#             with st.spinner("Scanning..."):
#                 result_img, results = detect_faces(
#                     camera_image.getvalue(),
#                     encodeListKnown,
#                     classNames
#                 )
#             show_results(result_img, results)

# # ─────────────────────────────────────────
# # TAB 2 — UPLOAD
# # ─────────────────────────────────────────
# with tab2:
#     st.markdown("#### Upload an image to mark attendance")
#     st.caption("Supported formats: JPG, JPEG, PNG")

#     uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#         if st.button("🔍 Detect Face", key="upload_detect", use_container_width=True):
#             with st.spinner("Scanning..."):
#                 result_img, results = detect_faces(
#                     uploaded_file.getvalue(),
#                     encodeListKnown,
#                     classNames
#                 )
#             show_results(result_img, results)

# # ══════════════════════════════════════════════════════════════
# #  ATTENDANCE LOG (bottom of page)
# # ══════════════════════════════════════════════════════════════
# st.divider()
# st.subheader("📋 Attendance Log")

# if os.path.exists('attendance.csv') and os.path.getsize('attendance.csv') > 0:
#     df = pd.read_csv('attendance.csv')
#     st.dataframe(df, use_container_width=True)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.download_button(
#             "⬇️ Download CSV",
#             df.to_csv(index=False),
#             "attendance.csv",
#             "text/csv",
#             use_container_width=True
#         )
#     with col2:
#         if st.button("🗑️ Clear Attendance", use_container_width=True):
#             os.remove('attendance.csv')
#             st.rerun()
# else:
#     st.info("No attendance marked yet. Use the camera or upload an image above.") 


import cv2
import face_recognition
import numpy as np
import os
import streamlit as st
from datetime import datetime
import pandas as pd

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Face Attendance System", page_icon="🎓", layout="centered")
st.title("🎓 Face Attendance System")
st.caption("Upload a photo to mark attendance automatically.")

# ─── Load dataset & encodings (cached — runs only once) ────────
@st.cache_resource
def load_encodings():
    path = 'Images_dataset'
    images, classNames = [], []

    for person in os.listdir(path):
        person_path = f'{path}/{person}'
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img = cv2.imread(f'{person_path}/{img_name}')
            if img is not None:
                images.append(img)
                classNames.append(person)

    encodeList = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img_rgb)
        if encodes:
            encodeList.append(encodes[0])

    return encodeList, classNames

# ─── Attendance helper ──────────────────────────────────────────
def markAttendance(name):
    file = 'attendance.csv'
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%d/%m/%Y')

    # Read existing names if file exists
    existing_names = []
    if os.path.exists(file):
        df = pd.read_csv(file)
        existing_names = df['Name'].tolist()

    if name not in existing_names:
        with open(file, 'a') as f:
            if not os.path.exists(file) or os.path.getsize(file) == 0:
                f.write("Name,Date,Time\n")
            f.write(f"{name},{date_str},{time_str}\n")
        return True  # newly marked
    return False     # already marked

# ─── Load encodings on startup ─────────────────────────────────
with st.spinner("Loading face encodings..."):
    encodeListKnown, classNames = load_encodings()

st.success(f"✅ Loaded {len(classNames)} people: {', '.join(set(classNames))}")
st.divider()

# ─── Upload section ────────────────────────────────────────────
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if st.button("🔍 Detect & Mark Attendance", use_container_width=True):
        with st.spinner("Scanning for faces..."):

            face_locs = face_recognition.face_locations(img_rgb, model='hog')
            face_encs = face_recognition.face_encodings(img_rgb, face_locs)

        if not face_encs:
            st.error("❌ No face detected in the image.")
        else:
            result_img = img.copy()
            results = []

            for encodeFace, faceLoc in zip(face_encs, face_locs):
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                if len(faceDis) == 0:
                    name = "UNKNOWN"
                    color = (0, 0, 255)
                else:
                    matchIndex = np.argmin(faceDis)
                    distance = faceDis[matchIndex]

                    if distance < 0.5:
                        name = classNames[matchIndex].upper()
                        color = (0, 255, 0)
                        newly_marked = markAttendance(classNames[matchIndex])
                        results.append({
                            "name": name,
                            "distance": round(distance, 4),
                            "status": "✅ Marked" if newly_marked else "⚠️ Already marked today"
                        })
                    else:
                        name = "UNKNOWN"
                        color = (0, 0, 255)
                        results.append({
                            "name": "UNKNOWN",
                            "distance": round(distance, 4),
                            "status": "❌ Not recognized / Imposter"
                        })

                # Draw box
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(result_img, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Show annotated image
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                     caption="Detection Result", use_column_width=True)

            # Show result cards
            for r in results:
                if "UNKNOWN" in r["name"]:
                    st.error(f"**{r['name']}** — {r['status']}  |  Distance: `{r['distance']}`")
                else:
                    st.success(f"**{r['name']}** — {r['status']}  |  Distance: `{r['distance']}`")

        st.divider()

# ─── Attendance table ──────────────────────────────────────────
st.subheader("📋 Attendance Log")
if os.path.exists('attendance.csv'):
    df = pd.read_csv('attendance.csv')
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False),
                       "attendance.csv", "text/csv")
else:
    st.info("No attendance marked yet.")