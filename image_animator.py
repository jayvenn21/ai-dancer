import numpy as np
import cv2
from PIL import Image

# Define body parts by keypoint indices (adjust if needed)
BODY_PARTS = {
    'head': [0, 1, 2, 3, 7],  # nose, eyes, ears (approximate)
    'left_upper_arm': [11, 13],
    'left_lower_arm': [13, 15],
    'right_upper_arm': [12, 14],
    'right_lower_arm': [14, 16],
    'left_upper_leg': [23, 25],
    'left_lower_leg': [25, 27],
    'right_upper_leg': [24, 26],
    'right_lower_leg': [26, 28],
    'torso': [11, 12, 23, 24]  # shoulders and hips
}

def crop_part(image_np, pt1, pt2, padding=20):
    x1, y1 = pt1
    x2, y2 = pt2
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)

    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(image_np.shape[1], xmax + padding)
    ymax = min(image_np.shape[0], ymax + padding)

    crop = image_np[ymin:ymax, xmin:xmax]
    return crop, (xmin, ymin)

def rotate_image(part_img, angle, pivot=None):
    (h, w) = part_img.shape[:2]
    if pivot is None:
        pivot = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(pivot, angle, 1.0)
    rotated = cv2.warpAffine(part_img, M, (w, h), borderMode=cv2.BORDER_TRANSPARENT)
    return rotated

def paste_part(canvas, part_img, position):
    x, y = position
    h, w = part_img.shape[:2]

    # Boundary check
    if y < 0 or x < 0 or y + h > canvas.shape[0] or x + w > canvas.shape[1]:
        # Clip part_img to fit inside canvas
        x1_src = 0
        y1_src = 0
        x2_src = w
        y2_src = h

        x1_dst = x
        y1_dst = y
        x2_dst = x + w
        y2_dst = y + h

        if x1_dst < 0:
            x1_src = -x1_dst
            x1_dst = 0
        if y1_dst < 0:
            y1_src = -y1_dst
            y1_dst = 0
        if x2_dst > canvas.shape[1]:
            x2_src -= (x2_dst - canvas.shape[1])
            x2_dst = canvas.shape[1]
        if y2_dst > canvas.shape[0]:
            y2_src -= (y2_dst - canvas.shape[0])
            y2_dst = canvas.shape[0]

        part_img = part_img[y1_src:y2_src, x1_src:x2_src]
        x, y, h, w = x1_dst, y1_dst, part_img.shape[0], part_img.shape[1]

    if part_img.shape[2] == 4:
        alpha_s = part_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            canvas[y:y+h, x:x+w, c] = (alpha_s * part_img[:, :, c] + alpha_l * canvas[y:y+h, x:x+w, c])
    else:
        canvas[y:y+h, x:x+w] = part_img
    return canvas

def generate_frames(image_np, keypoints, beat_times, fps=30, duration=10):
    frames = []
    beat_frames = set(int(bt * fps) for bt in beat_times)
    total_frames = int(fps * duration)

    # Crop body parts once
    parts = {}

    # Torso
    torso_pts = [keypoints[i] for i in BODY_PARTS['torso']]
    xs, ys = zip(*torso_pts)
    torso_crop, torso_pos = crop_part(image_np, (min(xs), min(ys)), (max(xs), max(ys)), padding=40)
    parts['torso'] = (torso_crop, torso_pos)

    # Head
    head_pts = [keypoints[i] for i in BODY_PARTS['head']]
    xs, ys = zip(*head_pts)
    head_crop, head_pos = crop_part(image_np, (min(xs), min(ys)), (max(xs), max(ys)), padding=20)
    parts['head'] = (head_crop, head_pos)

    limbs = ['left_upper_arm', 'left_lower_arm', 'right_upper_arm', 'right_lower_arm', 'left_upper_leg', 'left_lower_leg', 'right_upper_leg', 'right_lower_leg']
    for limb in limbs:
        pts = [keypoints[i] for i in BODY_PARTS[limb]]
        crop, pos = crop_part(image_np, pts[0], pts[1], padding=15)
        parts[limb] = (crop, pos)

    for i in range(total_frames):
        canvas = np.zeros_like(image_np)

        # Angles on beat for simple animation
        if i in beat_frames:
            angle_head = 5 * np.sin(i * 0.3)
            angle_arm = 15 * np.sin(i * 0.5)
            angle_leg = 10 * np.sin(i * 0.4)
        else:
            angle_head = 0
            angle_arm = 0
            angle_leg = 0

        # Paste torso (no rotation)
        torso_img, torso_pos = parts['torso']
        canvas = paste_part(canvas, torso_img, torso_pos)

        # Paste head (rotate around center)
        head_img, head_pos = parts['head']
        head_rot = rotate_image(head_img, angle_head)
        canvas = paste_part(canvas, head_rot, head_pos)

        # Arms
        for limb in ['left_upper_arm', 'left_lower_arm']:
            limb_img, limb_pos = parts[limb]
            limb_rot = rotate_image(limb_img, angle_arm)
            canvas = paste_part(canvas, limb_rot, limb_pos)

        for limb in ['right_upper_arm', 'right_lower_arm']:
            limb_img, limb_pos = parts[limb]
            limb_rot = rotate_image(limb_img, -angle_arm)
            canvas = paste_part(canvas, limb_rot, limb_pos)

        # Legs
        for limb in ['left_upper_leg', 'left_lower_leg']:
            limb_img, limb_pos = parts[limb]
            limb_rot = rotate_image(limb_img, angle_leg)
            canvas = paste_part(canvas, limb_rot, limb_pos)

        for limb in ['right_upper_leg', 'right_lower_leg']:
            limb_img, limb_pos = parts[limb]
            limb_rot = rotate_image(limb_img, -angle_leg)
            canvas = paste_part(canvas, limb_rot, limb_pos)

        frames.append(canvas)

    return frames
