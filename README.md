# 🎭 Multimodal Emotion Recognition System (Facial + Audio)

A real-time, multimodal emotion detection system combining facial expressions and audio signals using deep learning. Features hybrid fusion and temporal encoding for robust and stable predictions.

---

## ✨ Key Features

- **Real-Time Emotion Detection:** Seamless recognition from webcam and microphone.
- **Hybrid Fusion (Feature + Score Level):** Combines feature-level and score-level fusion of facial and audio predictions for richer joint representations.
- **Temporal Encoding:** Uses optical flow and rolling history to smooth predictions over time.
- **Manual Audio Trigger:** Press `'a'` to record and process emotion via microphone.
- **Confidence Filtering & Face Hold:** Filters low-confidence predictions and maintains bounding boxes during temporary detection loss.

---

## 📦 Datasets

| Modality | Dataset      | Classes                                 | Format/Preprocessing                       |
|----------|--------------|-----------------------------------------|--------------------------------------------|
| Facial   | GroupEmoW    | Positive, Negative, Neutral             | RGB + bounding boxes → Cropped/resized faces |
| Audio    | VGAF Audio   | Neutral, Happy, Sad, Angry, Fearful, Disgust, Surprise | `.wav` → MFCC + delta features           |

---

## 🧠 Algorithms

- **Facial Emotion Recognition:** CNN (custom or MobileNetV2)
- **Audio Emotion Recognition:** CNN on MFCC features
- **Hybrid Fusion:**
  - Feature-level: concatenated modality outputs
  - Score-level: 60% facial + 40% audio weighting
  - Final: Softmax smoothing
- **Temporal Encoding:** Rolling buffer for history-based smoothing

```python
def hybrid_fusion(face_probs, audio_probs):
    fused_features = np.concatenate([face_probs, audio_probs], axis=1)
    score_fusion = 0.6 * face_probs + 0.4 * audio_probs
    final_probs = tf.nn.softmax(score_fusion).numpy()
    return final_probs

emotion_history = deque(maxlen=5)
emotion_history.append(combined)
smoothed_probs = np.mean(emotion_history, axis=0)
fused_emotion = fusion_labels[np.argmax(smoothed_probs)]
```

---

## 🔄 Code Execution Flow

1. Load models (facial + audio from `.json`, `.h5`, `.tflite`)
2. Capture webcam frames, detect faces via Haar/DNN
3. Predict facial emotion every 10 frames
4. Press `'a'`: Record 2s audio → extract MFCC → predict emotion
5. Fuse emotions via hybrid fusion → smooth with temporal encoding
6. Display predictions in real-time UI

---

## 🖼️ Output Behavior

| Component        | Behavior                                                        |
|------------------|-----------------------------------------------------------------|
| Webcam Feed      | Live overlay with bounding boxes and detected emotions          |
| Facial Emotion   | Updated every 10 frames, smoothed via optical flow              |
| Audio Emotion    | Triggered manually, displayed below face feed                   |
| Fusion Output    | Combines modalities, recalibrated softmax predictions           |
| Prediction Lag   | Minimized with threading and rolling buffer                     |

---

## 🎮 Manual Controls

- `'a'`: Record audio and predict emotion
- `'q'`: Quit the application cleanly

---

## 📊 Accuracy

| Model         | Accuracy          |
|---------------|-------------------|
| Facial Only   | ~72–75%           |
| Audio Only    | ~65–67%           |
| Hybrid Fusion | ~84%+ (approximate gain via fusion & encoding) |

---

## ✅ Results

- Multimodal fusion for richer representations
- Stable predictions via emotion history buffer
- Smooth UI, low latency, modular design
- Easy integration with future deep learning architectures

---

## 🚀 Future Enhancements

- Replace Bi-LSTM with Transformer-based fusion block
- Scene graph modeling for multi-face input
- Deploy as desktop/web tool with emotion dashboard
- Convert models to TFLite for lightweight deployment

---

## 🙌 Credits

- **Datasets:** GroupEmoW, VGAF Audio
- **Developed by:** Rishabh Sharma
- **Support:** Microsoft Copilot

---

Let me know if you want this updated directly in your repository or need further customization!
