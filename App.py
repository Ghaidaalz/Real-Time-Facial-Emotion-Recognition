import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

@st.cache_resource
def load_emotion_models():
    emotion_vgg_model = tf.keras.models.load_model('emotion_vgg_model.h5')
    emotion_cnn_model = tf.keras.models.load_model('emotion_tuned_model.h5')  
    return emotion_vgg_model, emotion_cnn_model

@st.cache_resource
def load_body_models():
    body_posture_cnn_model = tf.keras.models.load_model('body_posture_cnn_model.h5')
    body_posture_vgg_model = tf.keras.models.load_model('body_posture_vgg_model.h5')
    body_posture_yolo_model = tf.keras.models.load_model('body_posture_yolo_model.h5')
    return body_posture_cnn_model, body_posture_vgg_model, body_posture_yolo_model

emotion_labels = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}
body_labels = {0: "Active", 1: "Lazy"}

emotion_vgg_model, emotion_cnn_model = load_emotion_models()
body_posture_cnn_model, body_posture_vgg_model, body_posture_yolo_model = load_body_models()

if 'run' not in st.session_state:
    st.session_state['run'] = False

if 'selected_emotion_model' not in st.session_state:
    st.session_state['selected_emotion_model'] = 'VGG Model'

if 'selected_body_model' not in st.session_state:
    st.session_state['selected_body_model'] = 'VGG Model'

def video_stream(emotion_model, body_model):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    frame_count = 0  
    emotion = ''  
    body_posture = ''  

    while st.session_state['run']:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        bodies = body_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=3, minSize=(100, 100))

        frame_count += 1
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            if frame_count % 5 == 0:  
                # Both VGG and CNN models expect 224x224 RGB images
                face_resized = cv2.resize(face, (224, 224))  
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB
                img_array = np.array(face_rgb) / 255.0
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                prediction = emotion_model.predict(img_array)
                emotion_label = np.argmax(prediction)
                emotion = emotion_labels[emotion_label] 

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, emotion, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
            )

        if len(bodies) == 0:
            st.warning("No body detected.")
        for (x, y, w, h) in bodies:
            body = frame[y:y+h, x:x+w]

            if frame_count % 5 == 0:  
                body_resized = cv2.resize(body, (224, 224))  # Body model expects 224x224 RGB images
                body_rgb = cv2.cvtColor(body_resized, cv2.COLOR_BGR2RGB)
                img_array = np.array(body_rgb) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = body_model.predict(img_array)
                body_label = np.argmax(prediction)
                body_posture = body_labels[body_label]  

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame, body_posture, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
            )

        stframe.image(frame, channels="BGR")

        if not st.session_state['run']:
            break

    cap.release()
    cv2.destroyAllWindows()

st.title("Real-Time Facial Emotion and Body Posture Recognition")

st.session_state['selected_emotion_model'] = st.selectbox("Select Emotion Model", ["VGG Model", "CNN Model"])
st.session_state['selected_body_model'] = st.selectbox("Select Body Posture Model", ["VGG Model", "CNN Model", "YOLO Model"])

if not st.session_state['run']:
    if st.button("Start Recognition"):
        st.session_state['run'] = True
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
