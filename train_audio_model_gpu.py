import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# ✅ GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("[INFO] GPU is available and configured for training.")
    except RuntimeError as e:
        print("[WARNING] GPU setup error:", e)
else:
    print("[INFO] No GPU found. Training will use CPU.")

# 🎭 Emotion mapping from RAVDESS filenames
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# 📁 Load dataset
X, y = [], []
audio_dir = "VGAF_AUDIO_DATASET"
for actor_folder in os.listdir(audio_dir):
    actor_path = os.path.join(audio_dir, actor_folder)
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            print(f"[INFO] Processing {file}...")
            emotion_code = file.split("-")[2]
            emotion = emotion_map.get(emotion_code)
            if emotion:
                file_path = os.path.join(actor_path, file)
                try:
                    features = extract_mfcc(file_path)
                    X.append(features)
                    y.append(emotion)
                except Exception as e:
                    print(f"[ERROR] Failed to process {file_path}: {e}")

# 🧠 Encode labels
X = np.array(X)
le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))

# 🔀 Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 🏗️ Build model
model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='softmax')  # 8 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🚀 Train model
print("[INFO] Training started...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
print("[INFO] Training completed.")

# 💾 Save model
model_json = model.to_json()
with open("audio_emotion_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("audio_emotion_model.weights.h5")
print("[INFO] Model saved as audio_emotion_model.json and audio_emotion_model.weights.h5")
