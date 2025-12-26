import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
model_path = '/absolute/path/to/pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)


cap = cv2.VideoCapture(0)
with PoseLandmarker.create_from_options(options) as landmarker:
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