import os
import cv2
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import threading
import mediapipe as mp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load audio emotion TFLite model
audio_interpreter = tf.lite.Interpreter(model_path="audio_emotion_model.tflite")
audio_interpreter.allocate_tensors()
audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()

# Load facial emotion TFLite model
face_interpreter = tf.lite.Interpreter(model_path="facial_emotion_model.tflite")
face_interpreter.allocate_tensors()
face_input_details = face_interpreter.get_input_details()
face_output_details = face_interpreter.get_output_details()

face_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
fusion_labels = list(face_labels.values())

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

SAMPLE_RATE = 22050
DURATION = 2
audio_probs = np.zeros((1, 7))
audio_thread_running = False


def extract_face_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1).astype(np.float32)
    return feature / 255.0


def predict_face_emotion_tflite(roi):
    face_interpreter.set_tensor(face_input_details[0]['index'], roi)
    face_interpreter.invoke()
    return face_interpreter.get_tensor(face_output_details[0]['index'])


def predict_audio_emotion_tflite(audio_input):
    audio_input = np.expand_dims(audio_input.astype(np.float32), axis=0)
    audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_input)
    audio_interpreter.invoke()
    return audio_interpreter.get_tensor(audio_output_details[0]['index'])[:, :7]


def record_audio_and_predict():
    global audio_probs, audio_thread_running
    if audio_thread_running:
        return
    audio_thread_running = True
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    audio = audio.flatten()
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        mfcc_processed = np.mean(mfcc.T, axis=0)
        audio_probs = predict_audio_emotion_tflite(mfcc_processed)
    except:
        audio_probs = np.zeros((1, 7))
    audio_thread_running = False


webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
threading.Thread(target=record_audio_and_predict).start()

while True:
    ret, frame = webcam.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    display_frame = frame.copy()

    detections = face_detector.process(frame_rgb)
    face_probs = np.zeros((1, 7))
    face_emotion = ""

    if detections.detections:
        detection = detections.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        x, y = max(x, 0), max(y, 0)
        roi_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_input = extract_face_features(roi_resized)
        face_probs = predict_face_emotion_tflite(roi_input)
        face_emotion = fusion_labels[np.argmax(face_probs)]
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(display_frame, f"Face: {face_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    mesh_results = face_mesh.process(frame_rgb)
    if mesh_results.multi_face_landmarks:
        for faceLms in mesh_results.multi_face_landmarks:
            ih, iw, _ = frame.shape
            for lm in faceLms.landmark:
                cx, cy = int(lm.x * iw), int(lm.y * ih)
                cv2.circle(display_frame, (cx, cy), 1, (0, 255, 255), -1)

    combined = 0.6 * face_probs + 0.4 * audio_probs
    fused_emotion = fusion_labels[np.argmax(combined)]

    cv2.putText(display_frame, f"Fused Emotion: {fused_emotion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    sidebar_x = 480
    cv2.rectangle(display_frame, (sidebar_x, 0), (640, 480), (0, 0, 0), -1)
    for idx, label in enumerate(fusion_labels):
        percent = int(combined[0][idx] * 100)
        text = f"{label.capitalize():<10} {percent} %"
        color = (255, 255, 255)
        if label == fused_emotion:
            color = (0, 0, 255)
        cv2.putText(display_frame, text, (sidebar_x + 10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(display_frame, "Status:", (sidebar_x + 10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(display_frame, "* Source: Webcam", (sidebar_x + 10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(display_frame, "* Player: Live", (sidebar_x + 10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(display_frame, "* Press 'a' to record audio", (sidebar_x + 10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

    cv2.imshow("Multimodal Emotion Detection (TFLite)", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        threading.Thread(target=record_audio_and_predict).start()
    elif key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
