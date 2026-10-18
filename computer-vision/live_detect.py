import cv2
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\prith\runs\detect\outputs\electronic_devices\weights\best.pt"
STREAM_URL = "http://192.168.0.101:4747/video"

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("Connecting to DroidCam...")
cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Could not open the video stream.")
    print("We need to verify the correct DroidCam stream URL.")
    raise SystemExit
print("Connected! Starting live detection. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Video stream stopped or returned no frame.")
        break

    results = model.predict(frame, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("AI Electronic Device Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()