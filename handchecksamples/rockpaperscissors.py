import cv2
import mediapipe as mp
import random
import time

# --- MediaPipe 設定 ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- カメラ起動 ---
cap = cv2.VideoCapture(0)

# --- 関数定義 ---
def get_finger_state(hand_landmarks):
    """各指が立っている(1)/曲がっている(0)かを返す"""
    finger_state = []
    tips = [4, 8, 12, 16, 20]
    mcp = [2, 5, 9, 13, 17]

    # 親指はx軸で判定（鏡像なので反転）
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
    """手の形をじゃんけんの形に変換"""
    if finger_state == [0, 0, 0, 0, 0]:
        return "Rock"
    elif finger_state == [1, 1, 1, 1, 1]:
        return "Paper"
    elif finger_state == [0, 1, 1, 0, 0]:
        return "Scissors"
    else:
        return None

def judge(player, cpu):
    """勝敗判定"""
    if player == cpu:
        return "Draw"
    if (player == "Rock" and cpu == "Scissors") or \
       (player == "Paper" and cpu == "Rock") or \
       (player == "Scissors" and cpu == "Paper"):
        return "You Win!"
    return "You Lose..."

# --- 変数 ---
player_hand = None
cpu_hand = None
result_text = ""
countdown_active = False
countdown_start = 0
countdown_value = 3

# --- メインループ ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # 手のサインを検出
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_state = get_finger_state(hand_landmarks)
            sign = get_hand_sign(finger_state)
            if sign:
                player_hand = sign

    # スペースキーでカウントダウン開始
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        countdown_active = True
        countdown_start = time.time()
        countdown_value = 3
        result_text = ""
        cpu_hand = None
    elif key == ord('q') or key == 27:  # 'q' または 'Esc'
        break

    # カウントダウン中の処理
    if countdown_active:
        elapsed = time.time() - countdown_start
        countdown_value = 3 - int(elapsed)

        if countdown_value > 0:
            cv2.putText(frame, str(countdown_value), (250, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
        else:
            # 判定開始
            countdown_active = False
            if player_hand:
                cpu_hand = random.choice(["Rock", "Paper", "Scissors"])
                result_text = judge(player_hand, cpu_hand)

    # --- 画面表示 ---
    cv2.putText(frame, "Press SPACE to Play JANKEN", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    if player_hand:
        cv2.putText(frame, f"You: {player_hand}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    if cpu_hand:
        cv2.putText(frame, f"CPU: {cpu_hand}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
    if result_text:
        cv2.putText(frame, result_text, (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

    cv2.imshow("Hand Janken Game", frame)

cap.release()
cv2.destroyAllWindows()