import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Callback function to handle results
def print_result(result, output_image, timestamp_ms):
    if result.hand_landmarks:
        print(f"Detected {len(result.hand_landmarks)} hands at {timestamp_ms}ms")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

cap = cv2.VideoCapture(0)
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Convert OpenCV BGR to MediaPipe RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Send to landmarker with a millisecond timestamp
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)
        
        cv2.imshow('MediaPipe Tasks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()