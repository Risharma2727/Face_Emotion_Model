import os
import cv2
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import threading
import mediapipe as mp
from tensorflow.keras.models import model_from_json, Sequential

# ✅ Use GPU 0 explicitly (optional if only one GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ✅ Confirm GPU visibility
gpus = tf.config.list_physical_devices('GPU')
print("[INFO] Available GPUs:", gpus)

# ✅ Set GPU memory growth to avoid crash
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("[ERROR] Could not set memory growth:", e)

# ✅ Log operations to see if they're running on GPU
tf.debugging.set_log_device_placement(True)

# 🎭 Load Facial Emotion Model
with open("facialemotionmodel.json", "r") as f:
    face_model_json = f.read()
face_model = model_from_json(face_model_json, custom_objects={"Sequential": Sequential})
face_model.load_weights("facialemotionmodel.h5")

# 🎙 Load Audio Emotion Model
with open("audio_emotion_model.json", "r") as f:
    audio_model_json = f.read()
audio_model = model_from_json(audio_model_json, custom_objects={"Sequential": Sequential})
audio_model.load_weights("audio_emotion_model.weights.h5")

# Emotion Labels
face_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
audio_labels = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}
fusion_labels = list(face_labels.values())

# Haar Cascade for face detection (used for emotion prediction ROI)
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Mediapipe Face Mesh for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SAMPLE_RATE = 22050
DURATION = 2

audio_probs = np.zeros((1, 7))  # global
prev_gray = None  # For optical flow

def extract_face_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

def record_audio_and_predict():
    global audio_probs
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    audio = audio.flatten()
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        mfcc_processed = np.mean(mfcc.T, axis=0)
        audio_input = np.expand_dims(mfcc_processed, axis=0)
        pred = audio_model.predict(audio_input)
        audio_probs = pred[:, :7]
    except Exception as e:
        print("Audio error:", e)
        audio_probs = np.zeros((1, 7))

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("[INFO] Press 'a' to record audio | 'q' to quit")

# Start audio thread initially
threading.Thread(target=record_audio_and_predict).start()

while True:
    ret, frame = webcam.read()
    if not ret:
        continue

    # Convert frame to grayscale for face detection and optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ===== Optical Flow Visualization =====
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        hsv = np.zeros_like(frame)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        display_frame = cv2.addWeighted(frame, 0.7, flow_rgb, 0.3, 0)
    else:
        display_frame = frame.copy()

    prev_gray = gray.copy()

    # ===== Facial Emotion Detection using Haar Cascade (for ROI) =====
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_probs = np.zeros((1, 7))
    face_emotion = ""

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = extract_face_features(roi)
        face_probs = face_model.predict(roi)
        face_emotion = face_labels[np.argmax(face_probs)]
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(display_frame, f"Face: {face_emotion}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # ===== Facial Landmark Detection with MediaPipe Face Mesh =====
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            h, w, _ = frame.shape
            for lm in faceLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Draw a small circle at each landmark point (yellow dots)
                cv2.circle(display_frame, (cx, cy), 1, (0, 255, 255), -1)

    # ===== Fusion Logic =====
    combined = 0.6 * face_probs + 0.4 * audio_probs
    fused_emotion = fusion_labels[np.argmax(combined)]
    cv2.putText(display_frame, f"Fused Emotion: {fused_emotion}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ===== Show Output =====
    cv2.imshow("Multimodal Emotion Detection + Optical Flow + Facial Landmarks", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        threading.Thread(target=record_audio_and_predict).start()
    elif key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()