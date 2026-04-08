# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# import os
# from datetime import datetime

# # Page Config
# st.set_page_config(page_title="Face Attendance System")
# st.title("📸 Face Recognition Attendance")

# # 1. Load Known Faces (Cached to prevent reloading)
# @st.cache_data
# def load_known_faces(path='Images_dataset'):
#     images = []
#     classNames = []
#     encodeList = []
    
#     if not os.path.exists(path):
#         return [], []

#     for person in os.listdir(path):
#         person_path = f'{path}/{person}'
#         if os.path.isdir(person_path):
#             for img_name in os.listdir(person_path):
#                 img = cv2.imread(f'{person_path}/{img_name}')
#                 img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 encodes = face_recognition.face_encodings(img_rgb)
#                 if encodes:
#                     encodeList.append(encodes[0])
#                     classNames.append(person.upper())
#     return encodeList, classNames

# # 2. Mark Attendance Logic
# def markAttendance(name):
#     with open('attendance.csv', 'a+') as f:
#         f.seek(0)
#         data = f.readlines()
#         names = [line.split(',')[0] for line in data]
#         if name not in names:
#             now = datetime.now()
#             f.writelines(f'\n{name},{now.strftime("%d/%m/%Y")},{now.strftime("%H:%M:%S")}')

# # Load data
# encodeListKnown, classNames = load_known_faces()

# # 3. UI - Image Upload
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert file to opencv image
#     file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
#     img = cv2.imdecode(file_bytes, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Process
#     with st.spinner('Scanning face...'):
#         face_locs = face_recognition.face_locations(img_rgb)
#         face_encodes = face_recognition.face_encodings(img_rgb, face_locs)

#     if not face_encodes:
#         st.error("No face detected!")
#     else:
#         for encodeFace, faceLoc in zip(face_encodes, face_locs):
#             faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            
#             if len(faceDis) > 0:
#                 matchIndex = np.argmin(faceDis)
                
#                 if faceDis[matchIndex] < 0.5:
#                     name = classNames[matchIndex]
#                     markAttendance(name)
#                     color = (0, 255, 0)
#                     st.success(f"✅ Detected: {name}")
#                 else:
#                     name = "UNKNOWN"
#                     color = (255, 0, 0)
#                     st.warning("❌ Imposter / Not Recognized")

#                 # Draw Box
#                 y1, x2, y2, x1 = faceLoc
#                 cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(img_rgb, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#         st.image(img_rgb, caption='Processed Image', use_column_width=True)



import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("📁 Image Upload Attendance System")

# Load and Encode known faces
@st.cache_data
def get_known_encodings(path='Images_dataset'):
    known_encodings = []
    known_names = []
    
    if not os.path.exists(path):
        st.error(f"Folder '{path}' not found!")
        return [], []

    for person in os.listdir(path):
        person_dir = os.path.join(path, person)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                img = cv2.imread(os.path.join(person_dir, img_name))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encodes = face_recognition.face_encodings(img_rgb)
                    if encodes:
                        known_encodings.append(encodes[0])
                        known_names.append(person.upper())
    return known_encodings, known_names

def markAttendance(name):
    # Ensure file exists
    if not os.path.isfile('attendance.csv'):
        with open('attendance.csv', 'w') as f:
            f.write('Name,Date,Time')
            
    with open('attendance.csv', 'r+') as f:
        data = f.readlines()
        name_list = [line.split(',')[0] for line in data]
        if name not in name_list:
            now = datetime.now()
            f.write(f'\n{name},{now.strftime("%d/%m/%Y")},{now.strftime("%H:%M:%S")}')
            return True
    return False

# Initialize
encodeListKnown, classNames = get_known_encodings()

uploaded_file = st.file_uploader("Upload an image for recognition", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Convert to OpenCV format
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with st.spinner('Processing...'):
        # Find faces in uploaded image
        face_locs = face_recognition.face_locations(img_display)
        face_encodes = face_recognition.face_encodings(img_display, face_locs)
        
        if not face_encodes:
            st.warning("No faces found in the uploaded image.")
        else:
            for encodeFace, faceLoc in zip(face_encodes, face_locs):
                # Compare
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                
                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    
                    # Threshold 0.5 (lower is better)
                    if faceDis[matchIndex] < 0.5:
                        name = classNames[matchIndex]
                        is_new = markAttendance(name)
                        color = (0, 255, 0)
                        st.success(f"✅ Match Found: {name} " + ("(Attendance Marked)" if is_new else "(Already Marked)"))
                    else:
                        name = "UNKNOWN"
                        color = (255, 0, 0)
                        st.error("❌ Not Recognized / Imposter")

                    # Draw Result
                    y1, x2, y2, x1 = faceLoc
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_display, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            st.image(img_display, caption="Recognition Result", use_column_width=True)
