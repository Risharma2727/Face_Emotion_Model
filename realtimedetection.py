import cv2
import numpy as np
import threading
import sounddevice as sd
import librosa
from collections import deque, Counter
from tensorflow.keras.models import model_from_json, Sequential
import tensorflow as tf
import time

# ✅ Load and compile face model
with open("facialemotionmodel.json") as f:
    face_model = model_from_json(f.read(), custom_objects={"Sequential": Sequential})
face_model.load_weights("facialemotionmodel.h5")
face_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Load and compile audio model
with open("audio_emotion_model.json") as f:
    audio_model = model_from_json(f.read(), custom_objects={"Sequential": Sequential})
audio_model.load_weights("audio_emotion_model.weights.h5")
audio_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🎭 Emotion labels
face_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
audio_labels = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}

# 🧠 Haar cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# 🎙️ Audio settings
SAMPLE_RATE = 22050
DURATION = 2
audio_emotion = "None"
audio_history = deque(maxlen=5)
running = True

def extract_face_features(image):
    image = image.reshape(1, 48, 48, 1) / 255.0
    return image

def record_audio():
    global audio_emotion, running
    if not running:
        return
    try:
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        if not running:
            return
        audio = audio.flatten()
        audio = audio / np.max(np.abs(audio))

        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        mfccs = librosa.feature.delta(mfccs)  # Add delta features
        mfccs_processed = np.mean(mfccs, axis=1)
        audio_features = np.expand_dims(mfccs_processed, axis=0)

        pred = audio_model.predict(audio_features)
        confidence = np.max(pred)
        if confidence > 0.6:  # Confidence threshold
            emotion = audio_labels[pred.argmax()]
            audio_history.append(emotion)
            audio_emotion = Counter(audio_history).most_common(1)[0][0]
            print(f"[AUDIO EMOTION] {audio_emotion} ({confidence:.2f})")
        else:
            print("[AUDIO EMOTION] Low confidence, ignored.")
    except Exception as e:
        print(f"[ERROR] Audio processing failed: {e}")

def start_audio_thread():
    threading.Thread(target=record_audio, daemon=True).start()

# 🎥 Webcam setup
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("[INFO] Press 'a' to record audio | 'q' to quit")

frame_count = 0
detect_interval = 5
predict_interval = 10
last_prediction_frame = 0
face_history = deque(maxlen=10)
stable_emotion = "neutral"

last_face = None
face_hold_frames = 15
face_hold_counter = 0

while running:
    ret, frame = webcam.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (480, 360))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_count += 1
    if frame_count % detect_interval == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) > 0:
            last_face = faces[0]
            face_hold_counter = face_hold_frames
    else:
        if face_hold_counter > 0 and last_face is not None:
            faces = [last_face]
            face_hold_counter -= 1
        else:
            faces = []

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (48, 48))
        img = extract_face_features(face_img)

        if frame_count - last_prediction_frame >= predict_interval:
            pred = face_model.predict(img)
            confidence = np.max(pred)
            if confidence > 0.6:
                face_emotion = face_labels[pred.argmax()]
                face_history.append(face_emotion)
                stable_emotion = Counter(face_history).most_common(1)[0][0]
            last_prediction_frame = frame_count

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, stable_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, f"Audio Emotion: {audio_emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Multimodal Emotion Detection", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('a'):
        print("[INFO] Recording audio...")
        start_audio_thread()
    elif key == ord('q'):
        running = False
        print("[INFO] Quitting...")
        break

webcam.release()
cv2.destroyAllWindows()
