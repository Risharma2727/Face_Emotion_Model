import tensorflow as tf
from tensorflow.keras.models import model_from_json, Sequential

# Load Keras model
with open("facialemotionmodel.json", "r") as f:
    model_json = f.read()
model = model_from_json(model_json, custom_objects={"Sequential": Sequential})
model.load_weights("facialemotionmodel.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional
tflite_model = converter.convert()

# Save TFLite model
with open("facial_emotion_model.tflite", "wb") as f:
    f.write(tflite_model)
