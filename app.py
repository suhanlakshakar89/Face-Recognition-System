import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace

st.set_page_config(page_title="Face Attendance", page_icon="🎓")
st.title("🎓 Face Recognition Attendance System")

DATASET_PATH = "Images_dataset"
ATTENDANCE_FILE = "attendance.csv"

# ── Mark Attendance ───────────────────────────────────────────
def markAttendance(name):
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

# ── Upload Image ──────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a photo to detect", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Save uploaded image temporarily
    temp_path = "temp_input.jpg"
    cv2.imwrite(temp_path, img_bgr)

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Face", type="primary"):
        with st.spinner("Detecting and matching face..."):
            try:
                # DeepFace.find compares uploaded image against all images in dataset
                results = DeepFace.find(
                    img_path=temp_path,
                    db_path=DATASET_PATH,
                    model_name="VGG-Face",
                    enforce_detection=True,
                    silent=True
                )

                # results is a list of DataFrames (one per face found)
                matched = False
                for df_result in results:
                    if df_result is not None and len(df_result) > 0:
                        best_match_path = df_result.iloc[0]["identity"]
                        # Extract person name from folder structure: Images_dataset/name/img.jpg
                        name = os.path.basename(os.path.dirname(best_match_path)).upper()

                        now = datetime.now()
                        st.success(f"✅ Detected Person: **{name}**")
                        st.write(f"📅 Date: {now.strftime('%d/%m/%Y')}  🕒 Time: {now.strftime('%H:%M:%S')}")

                        newly_marked = markAttendance(name)
                        if newly_marked:
                            st.info("📌 Attendance Marked!")
                        else:
                            st.warning("⚠️ Already marked today.")

                        matched = True
                        break

                if not matched:
                    st.error("❌ Imposter / Not Recognized")

            except ValueError as e:
                if "Face could not be detected" in str(e):
                    st.error("❌ No face detected in the image.")
                else:
                    st.error(f"❌ No match found — Imposter / Not Recognized")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ── Attendance Log ────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Attendance Log")
if os.path.exists(ATTENDANCE_FILE):
    df = pd.read_csv(ATTENDANCE_FILE)
    if df.empty:
        st.info("No records yet.")
    else:
        st.dataframe(df, use_container_width=True)
else:
    st.info("No records yet.")
