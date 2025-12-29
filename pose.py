import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = [
    # Facial landmarks (Eyes, Nose, Mouth)
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    # Legs
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

latest_result = None

def result_callback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """Callback function to grab results from the landmarker."""
    global latest_result
    latest_result = result
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=VisionRunningMode.LIVE_STREAM, result_callback=result_callback,min_pose_detection_confidence=0.5,min_pose_presence_confidence=0.5)

print("--- Start Webcam ---")
cap = cv2.VideoCapture(0)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # 1. Prepare image for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # 2. Send to detector (timestamp must be in milliseconds and increasing)
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_result is not None:
            # 3. Draw Results manually using OpenCV
            if latest_result.pose_landmarks:
                h, w, _ = frame.shape
                
                # The result is a list of lists (one list per person detected)
            for pose_landmarks in latest_result.pose_landmarks:

                # DRAW LINES with visibility check
                for start_idx, end_idx in POSE_CONNECTIONS:
                    start_lm = pose_landmarks[start_idx]
                    end_lm = pose_landmarks[end_idx]
                        
                    # Only draw if both points are likely visible (>50% confidence)
                    if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                        p1 = (int(start_lm.x * w), int(start_lm.y * h))
                        p2 = (int(end_lm.x * w), int(end_lm.y * h))
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)

                # DRAW POINTS
                for lm in pose_landmarks:
                    if lm.visibility > 0.5:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 0), -1)

        # 4. Display
        cv2.imshow('MediaPipe Pose - Modern Tasks API', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()