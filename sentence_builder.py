import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
from gtts import gTTS
from playsound import playsound
import uuid
import os

# ==============================
# LOAD STATIC MODEL
# ==============================
static_model = joblib.load("isl_alphabet_model.pkl")
STATIC_LABELS = [chr(i) for i in range(65, 91)]  # A-Z

# ==============================
# MEDIAPIPE HANDS
# ==============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==============================
# STATE VARIABLES
# ==============================
prev_keypoints = None
motion_threshold = 0.05

stable_count = 0
required_frames = 8

can_accept = True
sentence = ""
display_sign = ""

# ⏱ Space logic
last_hand_time = time.time()
SPACE_DELAY = 1.2
space_added = False

# ==============================
# ONLINE TTS
# ==============================
def speak_sentence_online(text):
    if text.strip() == "":
        return
    filename = f"tts_{uuid.uuid4()}.mp3"
    gTTS(text=text, lang="en").save(filename)
    playsound(filename)
    os.remove(filename)

# ==============================
# EXTRACT 126 KEYPOINTS (SAFE)
# ==============================
def extract_keypoints(results):
    left = np.zeros(63)
    right = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            points = []
            for lm in hand_landmarks.landmark:
                points.extend([lm.x, lm.y, lm.z])

            if label == "Left":
                left = np.array(points)
            else:
                right = np.array(points)

    return np.concatenate([left, right])  # ALWAYS 126

def backspace_sentence(text):
    if not text:
        return text

    # Remove trailing space first
    if text.endswith(" "):
        return text[:-1]

    # Remove last word if exists
    parts = text.rstrip().split(" ")
    if len(parts) > 1:
        return " ".join(parts[:-1]) + " "
    
    # Else remove last character
    return text[:-1]


# ==============================
# WEBCAM
# ==============================
cap = cv2.VideoCapture(0)
print("🎥 Webcam started | Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    # ==============================
    # NO HAND DETECTED
    # ==============================
    if not results.multi_hand_landmarks:
        prev_keypoints = None
        stable_count = 0
        can_accept = True
        display_sign = ""

        # ⏱ Add space ONLY after pause
        if time.time() - last_hand_time > SPACE_DELAY:
            if sentence and not sentence.endswith(" ") and not space_added:
                sentence += " "
                space_added = True

    # ==============================
    # HAND DETECTED
    # ==============================
    else:
        last_hand_time = time.time()
        space_added = False

        keypoints = extract_keypoints(results)

        motion = 0
        if prev_keypoints is not None:
            motion = np.linalg.norm(keypoints - prev_keypoints)
        prev_keypoints = keypoints

        # ==============================
        # STATIC LETTER LOGIC
        # ==============================
        if motion < motion_threshold:
            X = keypoints.reshape(1, -1)
            pred = static_model.predict(X)[0]

            if 0 <= pred < len(STATIC_LABELS):
                letter = STATIC_LABELS[pred]
            else:
                stable_count = 0
                continue

            stable_count += 1

            if stable_count >= required_frames:
                display_sign = letter

                if can_accept:
                    sentence += letter
                    can_accept = False

                stable_count = 0
        else:
            stable_count = 0
            can_accept = True

    # ==============================
    # UI
    # ==============================
    cv2.putText(frame, f"Sign: {display_sign}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.rectangle(frame, (20, 95), (620, 145), (0, 0, 0), -1)
    cv2.putText(frame, f"Sentence: {sentence[-40:]}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("ISL Sentence Builder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):   # 🔊 Speak
        speak_sentence_online(sentence)
    elif key == ord('c'):   # 🧹 Clear
        sentence = ""
    elif key == ord('b'):   # ⌫ Backspace last element
        sentence = backspace_sentence(sentence)

cap.release()
cv2.destroyAllWindows()
