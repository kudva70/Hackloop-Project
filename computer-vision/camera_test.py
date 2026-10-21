
import cv2

for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)

    if not cap.isOpened():
        print(f"Camera {index}: unavailable")
        cap.release()
        continue

    ret, frame = cap.read()

    if ret and frame is not None:
        print(f"Camera {index}: WORKING - {frame.shape}")
        cv2.imshow(f"Camera {index}", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    else:
        print(f"Camera {index}: opened, but no video frame")

    cap.release()

cv2.destroyAllWindows()
