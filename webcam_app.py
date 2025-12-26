import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# =========================
# Configuration
# =========================
IMG_SIZE = 48
PRED_WINDOW = 10
CONF_THRESHOLD = 0.5

# =========================
# Prediction buffer
# =========================
predictions = deque(maxlen=PRED_WINDOW)

# =========================
# Load trained model
# =========================
model = load_model("model/emotion_model.h5")

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]

# =========================
# Load face detector
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# Open webcam
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# =========================
# Main loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)  # ignore tiny faces
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        # Preprocess face
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # -------------------------
        # Predict emotion
        # -------------------------
        probs = model.predict(face, verbose=0)[0]
        predictions.append(probs)

        # Temporal smoothing
        avg_probs = np.mean(predictions, axis=0)
        confidence = np.max(avg_probs)
        emotion = emotion_labels[np.argmax(avg_probs)]

        if confidence < CONF_THRESHOLD:
            emotion_text = "Detecting..."
        else:
            emotion_text = f"{emotion} ({confidence:.2f})"

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            emotion_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Recognition", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
