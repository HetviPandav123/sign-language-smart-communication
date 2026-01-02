import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ================= CONFIG =================
ACTION = "thank_you"  # change to "thank_you"
SAMPLES = 200
FRAMES = 30
SAVE_PATH = f"dynamic_dataset/{ACTION}"
os.makedirs(SAVE_PATH, exist_ok=True)

# =========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

def extract_landmarks(results):
    data = []
    for i in range(2):
        if results.multi_hand_landmarks and i < len(results.multi_hand_landmarks):
            for lm in results.multi_hand_landmarks[i].landmark:
                data.extend([lm.x, lm.y, lm.z])
        else:
            data.extend([0] * 63)
    return data

sample_count = 0

while cap.isOpened() and sample_count < SAMPLES:
    print(f"\nPrepare for sample {sample_count+1}")
    time.sleep(1)

    sequence = []

    for frame_num in range(FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks = extract_landmarks(results)
        sequence.append(landmarks)

        cv2.putText(
            frame,
            f"Recording {ACTION}: {frame_num+1}/{FRAMES}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Collecting Dynamic Data", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    np.save(f"{SAVE_PATH}/{sample_count}.npy", np.array(sequence))
    sample_count += 1

print("✅ Dynamic data collection complete")
cap.release()
cv2.destroyAllWindows()
