import cv2
import numpy as np
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path).convert("RGBA")
    return np.array(img)

def shift_point(point, dx, dy):
    return (point[0] + dx, point[1] + dy)

def generate_frames(image, base_keypoints, beat_times, fps=30, duration=10):
    frames = []
    beat_frames = set(int(bt * fps) for bt in beat_times)
    total_frames = int(fps * duration)

    for i in range(total_frames):
        frame_img = image.copy()

        # Simple "dance move": shift shoulders and elbows on beat
        if i in beat_frames:
            dx = 10
            dy = 5
        else:
            dx = 0
            dy = 0

        # Copy image so we can draw pose (for demo purposes)
        img_pil = Image.fromarray(frame_img)

        # For demo: draw circles on shoulders and elbows and shift on beats
        # Mediapipe keypoints: 11 = left shoulder, 12 = right shoulder, 13 = left elbow, 14 = right elbow
        points_to_shift = [11, 12, 13, 14]

        # Draw original points
        for idx, (x, y) in enumerate(base_keypoints):
            color = (0, 255, 0)  # green
            radius = 5
            # On beat, shift some keypoints
            if idx in points_to_shift and i in beat_frames:
                x, y = shift_point((x, y), dx if idx % 2 == 0 else -dx, dy)
            img_pil = draw_circle(img_pil, (x, y), radius, color)

        frames.append(np.array(img_pil.convert("RGB")))
    return frames

def draw_circle(img, center, radius, color):
    cv_img = np.array(img)
    cv_img = cv2.circle(cv_img, center, radius, color, -1)
    return Image.fromarray(cv_img)
