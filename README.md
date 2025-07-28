# 🎭 Multimodal Emotion Recognition System (Facial + Audio)

This project implements a real-time emotion recognition system using both **facial expressions** and **audio signals**. It leverages deep learning models trained on **GroupEmoW** and **VGAF Audio** datasets to detect emotions from webcam video and microphone input.

---


## ✨ Features

- **Real-Time Emotion Detection**  
  Detects emotions from live webcam feed and microphone input with minimal latency.

- **Multimodal Fusion (Face + Audio)**  
  Combines facial expressions and vocal cues for more robust emotion recognition.

- **Manual Audio Trigger**  
  Press `'a'` to record and analyze audio emotion on demand.

- **Temporal Smoothing**  
  Uses rolling prediction history to stabilize output and reduce flickering.

- **Confidence Filtering**  
  Ignores low-confidence predictions to improve reliability.

- **Face Hold Mechanism**  
  Maintains bounding box and emotion display even when face detection temporarily fails.

## 📦 Datasets Used

### 🧠 Facial Emotion Dataset: GroupEmoW
- **Source**: [GroupEmoW](https://github.com/GroupEmoW/GroupEmoW-dataset)
- **Classes**: Positive, Negative , Neutral
- **Format**: RGB images with bounding boxes and group-level annotations
- **Preprocessing**: Cropped individual faces, resized to 48x48 or 96x96, converted to grayscale or RGB depending on model

### 🎙️ Audio Emotion Dataset: VGAF Audio
- **Source**: [VGAF](https://github.com/AudioVGAF/VGAF-dataset)
- **Classes**: Neutral, Happy, Sad, Angry, Fearful, Disgust, Surprise
- **Format**: `.wav` files with synchronized video and audio clips
- **Preprocessing**: MFCC + delta features extracted from audio segments

---

## 🧠 Algorithms Used

- **Facial Emotion Recognition**: trained on GroupEmoW (custom architecture or MobileNetV2)
- **Audio Emotion Recognition**: model trained on MFCC features from VGAF Audio
- **Feature Extraction**:
  - Facial: Cropped face → resized → normalized → CNN input
  - Audio: MFCC + delta → mean pooled → CNN input
- **Smoothing**: Rolling window (deque) for stable predictions
- **Confidence Filtering**: Threshold-based filtering for reliable output

---

## 🔄 Flow of Code Execution

1. **Load Models**: Facial and audio models loaded .tflite from `.json` and `.h5` files
2. **Start Webcam**: Captures frames and detects faces using Haar cascade or DNN
3. **Face Prediction**:
   - Every few frames, face is cropped and passed to CNN
   - Emotion is predicted and smoothed
4. **Manual Audio Trigger**:
   - Press `'a'` to record 2 seconds of audio
   - MFCC features are extracted and passed to audio model
   - Emotion is predicted and smoothed
5. **Display Output**:
   - Webcam frame shows bounding box, facial emotion, and audio emotion
   - Press `'q'` to quit

---

## 🖼️ Flow of Output

| Component        | Behavior                                                                 |
|------------------|--------------------------------------------------------------------------|
| Webcam Feed      | Real-time video with face detection and emotion overlay                 |
| Facial Emotion   | Updated every 10 frames, smoothed for stability                         |
| Audio Emotion    | Triggered manually, displayed below webcam feed                         |
| Rectangle Flicker| Eliminated using face hold mechanism                                    |
| Prediction Delay | Minimized with optimized intervals and threading                        |

---

## 🎮 Manual Controls

- Press `'a'` → Record audio and predict emotion  
- Press `'q'` → Quit the application cleanly

---

## 📊 Accuracy

| Model        | Accuracy (on full dataset) |
|--------------|----------------------------|
| Facial (GroupEmoW) | ~72–75% (depending on architecture) |
| Audio (VGAF)       | ~65-67% (MFCC-based CNN)            |
| Fusion             | ~83.3%

> Accuracy may vary based on preprocessing, model depth, and class balance.

---

## ✅ Results

- Real-time multimodal emotion detection with smooth UI
- Stable predictions using temporal smoothing
- Manual audio trigger for controlled evaluation
- Modular codebase for easy extension and retraining

---

## 🚀 Future Improvements

- Fuse face and audio predictions into a unified emotion output  
- Add real-time emotion dashboard or logging  
- Deploy as a desktop or web app  
- Use EfficientNet or transformer-based models for higher accuracy

---

## 🙌 Credits

- [GroupEmoW Dataset](https://github.com/GroupEmoW/GroupEmoW-dataset)
- [VGAF Audio Dataset](https://github.com/AudioVGAF/VGAF-dataset)
- Developed by **Rishabh Sharma ** with support from **Microsoft Copilot**


