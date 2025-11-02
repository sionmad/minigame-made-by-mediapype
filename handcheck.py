import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===== モデルファイルのパス =====
MODEL_PATH = r"C:\Users\asita\Downloads\minigame-made-by-mediapype-1\models\hand_landmarker.task"

# ===== HandLandmarkerの設定 =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

detector = HandLandmarker.create_from_options(options)

# ===== カメラの設定 =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCVはBGRなのでRGBに変換して左右反転
    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # MediaPipe専用形式に変換
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # 手の検出
    result = detector.detect(mp_image)

    # 結果を描画
    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            for landmark in landmarks:
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2
