import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Finger tips landmarks
finger_tips = [4, 8, 12, 16, 20]

# For webcam input:
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks, hand_label):
    count = 0

    # Thumb
    if hand_label == "Right":
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            count += 1
    else:
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            count += 1

    # Fingers (index to pinky)
    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1

    return count

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Flip image for selfie-view and convert to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process hands
    results = hands.process(image_rgb)

    total_fingers = 0
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_handedness.classification[0].label  # "Right" or "Left"
            total_fingers += count_fingers(hand_landmarks, hand_label)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Determine even or odd
    if total_fingers > 0:
        even_odd = "Even" if total_fingers % 2 == 0 else "Odd"
    else:
        even_odd = "No fingers"

    # Display finger count and even/odd
    cv2.putText(image, f'Fingers: {total_fingers}', (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(image, f'{even_odd}', (50, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    cv2.imshow('Finger Counter', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
