import numpy as np
import tensorflow as tf

# Load test data
X_test = np.load("X_face_test.npy")  # shape: (N, 48, 48, 1)
y_test = np.load("y_face_test.npy")  # shape: (N,)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="facial_emotion_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Inference function
def predict_single_sample(sample):
    sample = sample.astype(np.float32)
    sample = np.expand_dims(sample, axis=0)  # shape: (1, 48, 48, 1)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output)

# Make predictions
predictions = np.array([predict_single_sample(x) for x in X_test])

# Compute accuracy
accuracy = np.mean(predictions == y_test)
print(f"Face Emotion Model Accuracy: {accuracy * 100:.2f}%")

