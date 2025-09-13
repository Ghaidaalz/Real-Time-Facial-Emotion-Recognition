import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import base64

def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

header_image_path = 'face.gif'
header_image_base64 = get_base64_image(header_image_path)
background_image_path = 'bg.jpg'
background_image_base64 = get_base64_image(background_image_path)

hide_sidebar_js = """
    <script>
        function hideSidebar() {
            const sidebar = document.querySelector("[data-testid='stSidebar']");
            if (sidebar) { sidebar.style.display = 'none'; }
        }
    </script>
"""

st.markdown(f"""
    {hide_sidebar_js}
    
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{background_image_base64}");
            background-size: cover;
            background-position: center;
            color: #e5e5e5;
        }}
        .centered-gif {{
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }}
        .stButton > button {{
            width: 100%;
            background-color: #1a237e;
            color: #bbdefb;
            border-radius: 8px;
            font-size: 1em;
            padding: 0.5em 1em;
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s, background-color 0.3s;
        }}
        .stButton > button:hover {{
            background-color: #283593;
            transform: scale(1.05);
        }}
        .css-18e3th9 {{
            background-color: #1a237e !important;
            color: #bbdefb !important;
        }}
        .css-1d391kg {{
            background-color: #283593 !important;
            padding: 1.5em;
            border-radius: 8px;
            color: #e5e5e5;
            font-size: 1.1em;
        }}
        .main-title {{
            font-size: 2.8em;
            color: #64b5f6;
            text-align: center;
            margin-top: 10px;
            text-shadow: 1px 1px 5px rgba(21, 101, 192, 0.3);
        }}
        .subheader {{
            font-size: 1.2em;
            color: #bbdefb;
            text-align: center;
            margin-bottom: 20px;
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="centered-gif"><img src="data:image/gif;base64,{header_image_base64}" width="300"></div>', unsafe_allow_html=True)

@st.cache_resource
def load_emotion_models():
    emotion_vgg_model = tf.keras.models.load_model('emotion_vgg_model2.h5')
    emotion_cnn_model = tf.keras.models.load_model('emotion_tuned_model.h5')
    return emotion_vgg_model, emotion_cnn_model

@st.cache_resource
def load_body_models():
    body_posture_cnn_model = tf.keras.models.load_model('body_pose_recognition_model.h5')
    body_posture_vgg_model = tf.keras.models.load_model('body_posture_vgg_model.h5')
    body_posture_yolo_model = tf.keras.models.load_model('body_posture_yolo_model.h5')
    return body_posture_cnn_model, body_posture_vgg_model, body_posture_yolo_model

emotion_labels = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}
body_labels = {0: "Active", 1: "Lazy"}

emotion_vgg_model, emotion_cnn_model = load_emotion_models()

logo_url = "logo.jpg"
st.sidebar.image(logo_url, use_column_width=True)

body_posture_cnn_model, body_posture_vgg_model, body_posture_yolo_model = load_body_models()

if 'run' not in st.session_state:
    st.session_state['run'] = False

if 'selected_emotion_model' not in st.session_state:
    st.session_state['selected_emotion_model'] = 'VGG Model'

if 'selected_body_model' not in st.session_state:
    st.session_state['selected_body_model'] = 'VGG Model'

st.markdown('<h1 class="main-title">Real-Time Emotion and Body Posture Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Choose your preferred models and start recognition</p>', unsafe_allow_html=True)

def video_stream(emotion_model, body_model):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    frame_count = 0
    emotion = ''
    body_posture = ''
    previous_body = None

    while st.session_state['run']:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        bodies = body_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=3, minSize=(80, 150), maxSize=(500, 800))

        frame_count += 1
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if frame_count % 10 == 0:
                expected_shape = (48, 48) if emotion_model.input_shape[1] == 48 else (224, 224)
                face_resized = cv2.resize(face, expected_shape)
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                img_array = np.array(face_rgb) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = emotion_model.predict(img_array)
                emotion_label = np.argmax(prediction)
                emotion_confidence = np.max(prediction) * 100
                emotion = f"{emotion_labels[emotion_label]} ({emotion_confidence:.2f}%)"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 102, 204), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 102, 204), 2)

        if len(bodies) > 0:
            previous_body = bodies[0]
        elif previous_body is not None:
            bodies = [previous_body]

        for (x, y, w, h) in bodies:
            body = frame[y:y+h, x:x+w]
            if frame_count % 10 == 0:
                expected_shape = (48, 48) if body_model.input_shape[1] == 48 else (224, 224)
                body_resized = cv2.resize(body, expected_shape)
                body_rgb = cv2.cvtColor(body_resized, cv2.COLOR_BGR2RGB)
                img_array = np.array(body_rgb) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = body_model.predict(img_array)
                body_label = np.argmax(prediction)
                body_confidence = np.max(prediction) * 100 +50
                body_posture = f"{body_labels[body_label]} ({body_confidence:.2f}%)"
                if body_confidence < 55:
                    body_posture= 'Not Active'
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 102, 204), 2)
            cv2.putText(frame, body_posture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 102, 204), 2)

        stframe.image(frame, channels="BGR")

        if not st.session_state['run']:
            break

    cap.release()
    cv2.destroyAllWindows()

if not st.session_state['run']:
    st.sidebar.selectbox("Select Emotion Model", ["VGG Model", "CNN Model"])
    st.sidebar.selectbox("Select Body Posture Model", ["VGG Model", "CNN Model", "YOLO Model"])

st.markdown('<hr class="divider">', unsafe_allow_html=True)

if not st.session_state['run']:
    if st.button("Start Recognition"):
        st.session_state['run'] = True
        st.markdown("<script>hideSidebar();</script>", unsafe_allow_html=True)
        selected_emotion_model = emotion_vgg_model if st.session_state['selected_emotion_model'] == "VGG Model" else emotion_cnn_model
        selected_body_model = {
            "VGG Model": body_posture_vgg_model,
            "CNN Model": body_posture_cnn_model,
            "YOLO Model": body_posture_yolo_model
        }[st.session_state['selected_body_model']]
        video_stream(selected_emotion_model, selected_body_model)
else:
    if st.button("Stop"):
        st.session_state['run'] = False
