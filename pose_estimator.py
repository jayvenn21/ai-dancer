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
        # Convert normalized landmarks to pixel coordinates
        keypoints = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        return keypoints, image

if __name__ == "__main__":
    keypoints, image = get_pose_landmarks("assets/input.jpg")
    if keypoints:
        for point in keypoints:
            cv2.circle(image, point, 5, (0,255,0), -1)
        cv2.imshow("Pose Landmarks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
