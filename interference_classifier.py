import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3

# Load trained model and label map
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
label_map = model_dict.get('label_map', {})  # auto label map from training

# Reverse map: {0: 'HII', 1: 'HOW IS LIFE', ...}
labels_dict = {v: k for k, v in label_map.items()}
reverse_labels_dict = {v: k for k, v in label_map.items()}

# Initialize MediaPipe and TTS engine
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
engine = pyttsx3.init()

last_prediction = ''

while True:
    data_aux = []
    x_, y_ = [], []
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
                data_aux.extend([lm.x, lm.y])

        if len(data_aux) == 42:
            prediction = model.predict([np.array(data_aux)])
            predicted_label = int(prediction[0])
            predicted_text = reverse_labels_dict.get(predicted_label, "Unknown")

            if predicted_text != last_prediction:
                last_prediction = predicted_text
                engine.say(predicted_text)
                engine.runAndWait()

            x1, y1 = int(min(x_) * W), int(min(y_) * H)
            x2, y2 = int(max(x_) * W), int(max(y_) * H)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_text, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
