import cv2
import mediapipe as mp

# MediaPipe modules
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,                 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    # If hands detected
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Identify Left / Right hand
            hand_label = results.multi_handedness[idx].classification[0].label

            # Wrist landmark for text position
            wrist = hand_landmarks.landmark[0]
            h, w, _ = frame.shape
            cx, cy = int(wrist.x * w), int(wrist.y * h)

            cv2.putText(
                frame,
                hand_label,
                (cx - 30, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    cv2.imshow("Two Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
