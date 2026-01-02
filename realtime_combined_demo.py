"""
import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model
from collections import deque

# ============================
# LOAD MODELS
# ============================
static_model = joblib.load("isl_alphabet_model.pkl")
dynamic_model = load_model("dynamic_sign_model.h5")

STATIC_LABELS = [chr(i) for i in range(65, 91)]   # A-Z
DYNAMIC_LABELS = ["hello", "thank_you"]

# ============================
# MEDIAPIPE SETUP
# ============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================
# DYNAMIC BUFFER
# ============================
SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)

# ============================
# STABILITY + MOTION
# ============================
prev_keypoints = None
motion_threshold = 0.03
stable_sign = ""
static_count = 0
dynamic_count = 0

# ============================
# KEYPOINT EXTRACTION (126)
# ============================
def extract_126_keypoints(results):
    left = np.zeros(63)
    right = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            if label == "Left":
                left = np.array(data)
            else:
                right = np.array(data)

    return np.concatenate([left, right])

# ============================
# WEBCAM
# ============================
cap = cv2.VideoCapture(0)
print("🎥 Webcam started | Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    keypoints = extract_126_keypoints(results)

    # ============================
    # MOTION DETECTION
    # ============================
    motion = 0
    if prev_keypoints is not None:
        motion = np.linalg.norm(keypoints - prev_keypoints)
    prev_keypoints = keypoints

    # ============================
    # STATIC MODE
    # ============================
    if motion < motion_threshold and np.count_nonzero(keypoints) > 0:
        X_static = keypoints.reshape(1, -1)
        pred = static_model.predict(X_static)[0]
        letter = STATIC_LABELS[pred]

        static_count += 1
        if static_count > 7:
            stable_sign = letter
            static_count = 0

    # ============================
    # DYNAMIC MODE
    # ============================
    else:
        sequence.append(keypoints)
        if len(sequence) == SEQUENCE_LENGTH:
            X_dynamic = np.expand_dims(sequence, axis=0)  # (1,30,126)
            preds = dynamic_model.predict(X_dynamic, verbose=0)[0]
            confidence = np.max(preds)

            if confidence > 0.85:
                dynamic_count += 1
                if dynamic_count > 5:
                    stable_sign = DYNAMIC_LABELS[np.argmax(preds)]
                    dynamic_count = 0

    # ============================
    # DISPLAY (ONLY THIS)
    # ============================
    cv2.putText(
        frame,
        f"Sign: {stable_sign}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
"""
"""
static working with empty
import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from collections import deque, Counter

# ==============================
# LOAD MODELS
# ==============================
static_model = joblib.load("isl_alphabet_model.pkl")
dynamic_model = load_model("dynamic_sign_model.h5")

# Labels
static_labels = [chr(i) for i in range(65, 91)]  # A-Z
dynamic_labels = ["hello", "thank_you"]          # Add more if needed

# ==============================
# MEDIAPIPE HANDS
# ==============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==============================
# SEQUENCE BUFFER FOR DYNAMIC
# ==============================
SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)
pred_buffer = deque(maxlen=5)  # For smoothing predictions

# ==============================
# HELPER FUNCTIONS
# ==============================
def extract_keypoints(results):
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            hand = []
            for lm in hand_landmarks.landmark:
                hand.extend([lm.x, lm.y, lm.z])
            if label == "Left":
                left_hand = np.array(hand)
            else:
                right_hand = np.array(hand)
    return np.concatenate([left_hand, right_hand])

def compute_motion(curr, prev):
    if curr is None or prev is None:
        return 0
    return np.mean(np.abs(curr - prev))

# ==============================
# WEBCAM
# ==============================
cap = cv2.VideoCapture(0)
prev_landmarks = None
last_sign = ""
motion_threshold = 0.008   # Adjust for sensitivity

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
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Extract landmarks
    landmarks = extract_keypoints(results)

    # ==========================
    # DETERMINE MOTION
    # ==========================
    motion = compute_motion(landmarks, prev_landmarks)
    prev_landmarks = landmarks.copy()

    # ==========================
    # STATIC vs DYNAMIC
    # ==========================
    if np.sum(landmarks) == 0:
        last_sign = ""  # No hands detected → show nothing
        sequence.clear()  # Reset dynamic sequence
        pred_buffer.clear()
    elif motion < motion_threshold:
        # --- Static sign ---
        X = landmarks.reshape(1, -1)
        pred = static_model.predict(X)[0]
        last_sign = static_labels[pred]
    else:
        # --- Dynamic sign ---
        sequence.append(landmarks)
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)  # (1,30,126)
            preds = dynamic_model.predict(input_data, verbose=0)[0]
            confidence = np.max(preds)
            predicted_label = dynamic_labels[np.argmax(preds)]
            if confidence > 0.85:
                pred_buffer.append(predicted_label)
                # Smooth predictions: take most common in buffer
                last_sign = Counter(pred_buffer).most_common(1)[0][0]

    # ==========================
    # DISPLAY
    # ==========================
    if last_sign != "":
        cv2.putText(frame, f"Sign: {last_sign}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("ISL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
"""

"""dynamic working without empty
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import joblib

# ==============================
# LOAD MODELS
# ==============================
static_model = joblib.load("isl_alphabet_model.pkl")
static_labels = [chr(i) for i in range(65, 91)]  # A-Z

dynamic_model = load_model("dynamic_sign_model.h5")
dynamic_labels = ["hello", "thank_you"]  # update if more

# ==============================
# MEDIAPIPE HANDS
# ==============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==============================
# SEQUENCE BUFFER FOR DYNAMIC SIGNS
# ==============================
SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)
confidence_threshold = 0.90  # higher for stability
motion_threshold = 0.02      # minimal movement ignored
last_dynamic_prediction = ""
prev_keypoints = None

# ==============================
# EXTRACT 126 KEYPOINTS
# ==============================
def extract_keypoints(results):
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            hand = []
            for lm in hand_landmarks.landmark:
                hand.extend([lm.x, lm.y, lm.z])
            if label == "Left":
                left_hand = np.array(hand)
            else:
                right_hand = np.array(hand)

    return np.concatenate([left_hand, right_hand])

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
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ==============================
    # STATIC SIGN DETECTION
    # ==============================
    static_landmarks_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_data = [lm.x for lm in hand_landmarks.landmark] + \
                        [lm.y for lm in hand_landmarks.landmark] + \
                        [lm.z for lm in hand_landmarks.landmark]
            static_landmarks_list.append(hand_data)

        final_landmarks = []
        if len(static_landmarks_list) == 1:
            final_landmarks = static_landmarks_list[0] + [0.0]*63
        elif len(static_landmarks_list) == 2:
            final_landmarks = static_landmarks_list[0] + static_landmarks_list[1]

        if len(final_landmarks) == 126:
            X_static = np.array(final_landmarks).reshape(1, -1)
            pred_static = static_model.predict(X_static)[0]
            static_sign = static_labels[pred_static]
        else:
            static_sign = ""
    else:
        static_sign = ""

    # ==============================
    # DYNAMIC SIGN DETECTION
    # ==============================
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    motion = 0
    if prev_keypoints is not None:
        motion = np.mean(np.abs(keypoints - prev_keypoints))
    prev_keypoints = keypoints

    dynamic_sign = last_dynamic_prediction
    if len(sequence) == SEQUENCE_LENGTH and motion > motion_threshold:
        X_dynamic = np.expand_dims(sequence, axis=0)
        preds = dynamic_model.predict(X_dynamic, verbose=0)[0]
        confidence = np.max(preds)
        predicted_label = dynamic_labels[np.argmax(preds)]
        if confidence > confidence_threshold:
            last_dynamic_prediction = predicted_label
            dynamic_sign = last_dynamic_prediction

    # ==============================
    # DISPLAY SIGN
    # ==============================
    display_sign = static_sign if static_sign else dynamic_sign
    cv2.putText(frame, f"Sign: {display_sign}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("ISL Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""



"""
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import joblib
import time

# ==============================
# LOAD MODELS
# ==============================
static_model = joblib.load("isl_alphabet_model.pkl")
static_labels = [chr(i) for i in range(65, 91)]  # A-Z

dynamic_model = load_model("dynamic_sign_model.h5")
dynamic_labels = ["hello", "thank_you"]  # add more if needed

# ==============================
# MEDIAPIPE HANDS
# ==============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==============================
# SEQUENCE BUFFER FOR DYNAMIC SIGNS
# ==============================
SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)
confidence_threshold = 0.90  # dynamic sign stability
motion_threshold = 0.2      # minimal movement ignored
prev_keypoints = None
confirmed_dynamic_sign = ""  # dynamic signs stick until changed

# ==============================
# STATIC SIGN TIMING
# ==============================
static_timer = 0.3
static_hold_time = 1.8  # seconds to confirm static sign
last_static_sign = ""

# ==============================
# TRACK DISPLAYED SIGN
# ==============================
display_sign = ""
display_type = None  # "dynamic" or "static"

# ==============================
# EXTRACT 126 KEYPOINTS
# ==============================
def extract_keypoints(results):
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            hand = []
            for lm in hand_landmarks.landmark:
                hand.extend([lm.x, lm.y, lm.z])
            if label == "Left":
                left_hand = np.array(hand)
            else:
                right_hand = np.array(hand)

    return np.concatenate([left_hand, right_hand])

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

    # Draw landmarks if any
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ==============================
    # NO HAND → show nothing
    # ==============================
    if not results.multi_hand_landmarks:
        display_sign = ""
        display_type = None
        sequence.clear()
        prev_keypoints = None
        static_timer = 0.2

    # ==============================
    # STATIC SIGN DETECTION
    # ==============================
    if results.multi_hand_landmarks:
        static_landmarks_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand_data = []
            for lm in hand_landmarks.landmark:
                hand_data.extend([lm.x, lm.y, lm.z])
            static_landmarks_list.append(hand_data)

        final_landmarks = []
        if len(static_landmarks_list) == 1:
            final_landmarks = static_landmarks_list[0] + [0.0]*63
        elif len(static_landmarks_list) == 2:
            final_landmarks = static_landmarks_list[0] + static_landmarks_list[1]

        if len(final_landmarks) == 126:
            X_static = np.array(final_landmarks).reshape(1, -1)
            pred_static = static_model.predict(X_static)[0]
            static_sign = static_labels[pred_static]

            # Timer logic for confirming static sign
            current_time = time.time()
            if static_sign == last_static_sign:
                if static_timer == 0.2:
                    static_timer = current_time
                elif (current_time - static_timer) >= static_hold_time:
                    # Confirmed static sign → show it if last displayed was dynamic or nothing
                    if display_type != "static":
                        display_sign = static_sign
                        display_type = "static"
            else:
                last_static_sign = static_sign
                static_timer = current_time

    # ==============================
    # DYNAMIC SIGN DETECTION
    # ==============================
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    motion = 0
    if prev_keypoints is not None:
        motion = np.mean(np.abs(keypoints - prev_keypoints))
    prev_keypoints = keypoints

    if results.multi_hand_landmarks and len(sequence) == SEQUENCE_LENGTH:
        if motion > motion_threshold:
            X_dynamic = np.expand_dims(sequence, axis=0)
            preds = dynamic_model.predict(X_dynamic, verbose=0)[0]
            confidence = np.max(preds)
            predicted_label = dynamic_labels[np.argmax(preds)]

            if confidence > confidence_threshold:
                if predicted_label != confirmed_dynamic_sign:
                    confirmed_dynamic_sign = predicted_label
                    # Immediately switch to dynamic sign
                    display_sign = confirmed_dynamic_sign
                    display_type = "dynamic"

    # ==============================
    # DISPLAY SIGN
    # ==============================
    cv2.putText(frame, f"Sign: {display_sign}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("ISL Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model
from collections import deque

# ============================
# LOAD MODELS
# ============================
static_model = joblib.load("isl_alphabet_model.pkl")
dynamic_model = load_model("dynamic_sign_model.h5")

STATIC_LABELS = [chr(i) for i in range(65, 91)]
DYNAMIC_LABELS = ["hello", "thank_you"]

# ============================
# MEDIAPIPE
# ============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================
# BUFFERS
# ============================
SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)

prev_keypoints = None
display_sign = ""

# ============================
# INTENT THRESHOLDS
# ============================
STATIC_MOTION_TH = 0.015
DYNAMIC_MOTION_TH = 0.04

STATIC_FRAMES = 6
DYNAMIC_FRAMES = 10

static_counter = 0
dynamic_counter = 0
current_intent = None  # "STATIC" | "DYNAMIC"

# ============================
# KEYPOINT EXTRACTION
# ============================
def extract_keypoints(results):
    left = np.zeros(63)
    right = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hl in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            data = []
            for lm in hl.landmark:
                data.extend([lm.x, lm.y, lm.z])
            if label == "Left":
                left = np.array(data)
            else:
                right = np.array(data)

    return np.concatenate([left, right])

# ============================
# WEBCAM
# ============================
cap = cv2.VideoCapture(0)
print("🎥 Webcam started | Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    # ============================
    # NO HAND → RESET
    # ============================
    if not results.multi_hand_landmarks:
        prev_keypoints = None
        sequence.clear()
        static_counter = dynamic_counter = 0
        current_intent = None
        display_sign = ""
        cv2.imshow("ISL Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # ============================
    # MOTION
    # ============================
    keypoints = extract_keypoints(results)

    motion = 0
    if prev_keypoints is not None:
        motion = np.linalg.norm(keypoints - prev_keypoints)
    prev_keypoints = keypoints

    # ============================
    # INTENT DETECTION
    # ============================
    if motion < STATIC_MOTION_TH:
        static_counter += 1
        dynamic_counter = 0
    elif motion > DYNAMIC_MOTION_TH:
        dynamic_counter += 1
        static_counter = 0
    else:
        static_counter = dynamic_counter = 0

    # ============================
    # STATIC INTENT
    # ============================
    if static_counter >= STATIC_FRAMES and current_intent != "STATIC":
        X = keypoints.reshape(1, -1)
        pred = static_model.predict(X)[0]
        display_sign = STATIC_LABELS[pred]
        current_intent = "STATIC"

    # ============================
    # DYNAMIC INTENT
    # ============================
    if dynamic_counter >= DYNAMIC_FRAMES:
        sequence.append(keypoints)
        if len(sequence) == SEQUENCE_LENGTH:
            X = np.expand_dims(sequence, axis=0)
            preds = dynamic_model.predict(X, verbose=0)[0]
            if np.max(preds) > 0.85:
                display_sign = DYNAMIC_LABELS[np.argmax(preds)]
                current_intent = "DYNAMIC"
                sequence.clear()

    # ============================
    # DISPLAY
    # ============================
    cv2.putText(frame, f"Sign: {display_sign}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("ISL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
