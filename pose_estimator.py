import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def get_pose_landmarks_3d(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print("No pose landmarks detected.")
            return None, None

        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape
        keypoints_2d = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        keypoints_3d = [(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks]

        return keypoints_2d, keypoints_3d
