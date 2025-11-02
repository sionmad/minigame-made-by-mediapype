import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ カメラが開けませんでした。")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ フレームを取得できません。")
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(5) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
