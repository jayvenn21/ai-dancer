import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def get_pose_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print("No pose landmarks detected.")
            return None

        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape
        keypoints = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        return keypoints, image
