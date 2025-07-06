import cv2
import numpy as np
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path).convert("RGBA")
    return np.array(img)

def shift_point(point, dx, dy):
    return (point[0] + dx, point[1] + dy)

def rotate_point(pivot, point, angle_deg):
    angle_rad = np.radians(angle_deg)
    ox, oy = pivot
    px, py = point
    qx = ox + np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy)
    qy = oy + np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy)
    return int(qx), int(qy)

def draw_circle(img, center, radius, color):
    cv_img = np.array(img)
    cv_img = cv2.circle(cv_img, center, radius, color, -1)
    return Image.fromarray(cv_img)

def generate_frames(image, base_keypoints, beat_times, fps=30, duration=10):
    frames = []
    beat_frames = set(int(bt * fps) for bt in beat_times)
    total_frames = int(fps * duration)

    for i in range(total_frames):
        keypoints = base_keypoints.copy()

        # ==== Animate entire image bounce ====
        dy = -10 if i in beat_frames else 0
        M = np.float32([[1, 0, 0], [0, 1, dy]])
        shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        img_pil = Image.fromarray(shifted_image)

        # ==== Animate arms ====
        if i in beat_frames:
            try:
                keypoints[13] = rotate_point(keypoints[11], keypoints[13], 15)
                keypoints[15] = rotate_point(keypoints[13], keypoints[15], 20)
                keypoints[14] = rotate_point(keypoints[12], keypoints[14], -15)
                keypoints[16] = rotate_point(keypoints[14], keypoints[16], -20)
                keypoints[0] = shift_point(keypoints[0], 0, dy)
            except IndexError:
                pass  # in case keypoints are missing

        for x, y in keypoints:
            img_pil = draw_circle(img_pil, (x, y), 4, (0, 255, 0))

        frames.append(np.array(img_pil.convert("RGB")))

    return frames
