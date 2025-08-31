import os
import cv2
import pickle #If you have data you don’t want to compute again or want to save something for later
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data_dir = './data'
data = []
labels = []


gesture_folders = sorted([folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))])
label_map = {gesture_name: idx for idx, gesture_name in enumerate(gesture_folders)}

print(f"📂 Detected gesture folders: {label_map}")

for label_name, label_index in label_map.items():
    folder_path = os.path.join(data_dir, label_name)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        img = cv2.imread(file_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            data_aux = []
            for lm in hand.landmark:
                data_aux.append(lm.x)
                data_aux.append(lm.y)
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(label_index)


with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'label_map': label_map}, f)

print(f"\nSaved {len(data)} samples to 'data.pickle'")
print("🧾 Label map saved inside the pickle file:")
print(label_map)
