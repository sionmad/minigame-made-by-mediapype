import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

def get_finger_state(hand_landmarks):
    """
    指が立っているか(1)・曲がっているか(0)を返す
    [thumb, index, middle, ring, pinky]
    """
    finger_state = []

    # 各指の先端と付け根
    tips = [4, 8, 12, 16, 20]
    mcp = [2, 5, 9, 13, 17]

    # 親指はx軸で判定（左右反転注意）
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[mcp[0]].x:
        finger_state.append(1)
    else:
        finger_state.append(0)

    # 他の4本はy軸で判定
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[mcp[i]].y:
            finger_state.append(1)
        else:
            finger_state.append(0)

    return finger_state

def get_hand_sign(finger_state):
    if finger_state == [0, 0, 0, 0, 0]:
        return "FIST"
    elif finger_state == [1, 1, 1, 1, 1]:
        return "OPENDHAND"
    elif finger_state == [0, 1, 1, 0, 0]:
        return "PEACESIGN"
    elif finger_state == [1, 0, 0, 0, 0]:
        return "THUMBUP"
    else:
        return "UNKNOWN"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_state = get_finger_state(hand_landmarks)
            sign = get_hand_sign(finger_state)
            cv2.putText(frame, sign, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Hand Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()