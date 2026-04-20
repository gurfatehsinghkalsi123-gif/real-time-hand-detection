import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
print("hand tracking started! press 'q' to quit.")

def detect_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 7, 11, 15, 19]
    extended = 0
    if abs(landmarks[tip_ids[0]].x - landmarks[pip_ids[0]].x) > 0.04:
        extended += 1
    for i in range(1, 5):
        if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
            extended += 1
    if extended >= 4:
        return "Open Hand"
    elif extended == 0:
        return "Fist"
    else:
        return "partially open"
while True:
    success, img = cap.read()
    if not success:
        break
    frame = cv2.flip(img, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    gesture = "No Hand Detected"
    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[0].classification[0].label
            gesture = detect_gesture(hand_landmarks)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingertip_ids = [4, 8, 12, 16, 20]
            for tip_id in fingertip_ids:
                lm = hand_landmarks.landmark[tip_id]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, str(tip_id), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, f"{hand_label}: {gesture}", (wrist_x - 50, wrist_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    status_color = (0, 255, 0) if gesture == "Open Hand" else (0, 0, 255) if gesture == "Fist" else (255, 255, 0)
    cv2.putText(frame, f"Status: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


          
