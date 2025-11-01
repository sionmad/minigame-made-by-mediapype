import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===== モデルファイルのパス =====
MODEL_PATH = "venv/Lib/site-packages/mediapipe/modules/hand_landmark/hand_landmarker.task"

# ===== HandLandmarkerの設定 =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,  # 検出する手の数
)

detector = HandLandmarker.create_from_options(options)

# ===== カメラの設定 =====
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # カメラ映像を左右反転＆RGB変換
    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # MediaPipeの画像形式に変換
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # 手の検出を実行
    detection_result = detector.detect(mp_image)

    # ===== 結果を描画 =====
    annotated_frame = frame.copy()

    if detection_result.hand_landmarks:
        for landmarks in detection_result.hand_landmarks:
            for lm in landmarks:
                h, w, _ = annotated_frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 0), -1)

    cv2.imshow('MediaPipe Hands (v0.10.x)', annotated_frame)

    # ===== 終了条件 =====
    # ウィンドウを閉じたら終了
    if cv2.getWindowProperty('MediaPipe Hands (v0.10.x)', cv2.WND_PROP_VISIBLE) < 1:
        break

    # ESC または q で終了
    key = cv2.waitKey(5) & 0xFF
    if key == 27 or key == ord('q'):
        break

# ===== 終了処理 =====
cap.release()
cv2.destroyAllWindows()
