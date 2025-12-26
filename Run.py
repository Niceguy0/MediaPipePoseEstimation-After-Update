import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
model_path = '/absolute/path/to/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Callback function to handle results
def print_result(result, output_image, timestamp_ms):
    if result.hand_landmarks:
        print(f"Detected {len(result.hand_landmarks)} hands at {timestamp_ms}ms")

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    mp.tasks.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      mp.tasks.pose.POSE_CONNECTIONS,
      mp.tasks.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

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
        
        cv2.imshow('MediaPipe Tasks', draw_landmarks_on_image(frame, landmarker.detect(mp_image)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()