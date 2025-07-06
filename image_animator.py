import numpy as np
import cv2
from PIL import Image

# Define body parts by keypoint indices (extended for better cropping)
BODY_PARTS = {
    'head': [0, 1, 2, 3, 7, 4, 5, 6],  # nose, eyes, ears, mouth corners
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

def crop_part(image_np, points, padding=20):
    # points is a list of tuples (x, y)
    xs, ys = zip(*points)
    xmin, xmax = max(min(xs) - padding, 0), min(max(xs) + padding, image_np.shape[1])
    ymin, ymax = max(min(ys) - padding, 0), min(max(ys) + padding, image_np.shape[0])
    crop = image_np[ymin:ymax, xmin:xmax]
    return crop, (xmin, ymin)

def rotate_image(part_img, angle, pivot):
    (h, w) = part_img.shape[:2]
    M = cv2.getRotationMatrix2D(pivot, angle, 1.0)
    rotated = cv2.warpAffine(part_img, M, (w, h), borderMode=cv2.BORDER_TRANSPARENT)
    return rotated

def paste_part(canvas, part_img, position):
    x, y = position
    h, w = part_img.shape[:2]

    # Check boundaries
    if y < 0 or x < 0 or y + h > canvas.shape[0] or x + w > canvas.shape[1]:
        # Skip parts that go out of bounds
        return canvas

    # Handle alpha blending if present
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

    # Pre-crop all parts once
    parts = {}

    # Torso
    torso_pts = [keypoints[i] for i in BODY_PARTS['torso']]
    torso_crop, torso_pos = crop_part(image_np, torso_pts, padding=40)
    parts['torso'] = (torso_crop, torso_pos)

    # Head
    head_pts = [keypoints[i] for i in BODY_PARTS['head']]
    head_crop, head_pos = crop_part(image_np, head_pts, padding=30)
    parts['head'] = (head_crop, head_pos)

    # Limbs
    limbs = [
        ('left_upper_arm', [11, 13]),
        ('left_lower_arm', [13, 15]),
        ('right_upper_arm', [12, 14]),
        ('right_lower_arm', [14, 16]),
        ('left_upper_leg', [23, 25]),
        ('left_lower_leg', [25, 27]),
        ('right_upper_leg', [24, 26]),
        ('right_lower_leg', [26, 28]),
    ]

    for limb_name, indices in limbs:
        pts = [keypoints[i] for i in indices]
        crop, pos = crop_part(image_np, pts, padding=30)
        parts[limb_name] = (crop, pos)

    # For smooth rotation, define a simple sine wave function for angles
    def angle_for_frame(i, max_angle):
        # oscillate angle between -max_angle and +max_angle smoothly over time
        return max_angle * np.sin(2 * np.pi * i / fps * 2)  # 2 Hz oscillation

    for i in range(total_frames):
        canvas = np.zeros_like(image_np)

        # Determine rotation angles
        on_beat = i in beat_frames

        # Head nods a little always, stronger on beat
        angle_head = angle_for_frame(i, 5) + (10 if on_beat else 0)

        # Arms swing, stronger on beat
        angle_upper_arm = angle_for_frame(i, 15) * (2 if on_beat else 1)
        angle_lower_arm = angle_for_frame(i, 10) * (2 if on_beat else 1)

        # Legs kick a little, stronger on beat
        angle_upper_leg = angle_for_frame(i, 10) * (2 if on_beat else 1)
        angle_lower_leg = angle_for_frame(i, 7) * (2 if on_beat else 1)

        # Paste torso (no rotation)
        torso_crop, torso_pos = parts['torso']
        canvas = paste_part(canvas, torso_crop, torso_pos)

        # Paste head rotated around bottom-center (approximate neck)
        head_crop, head_pos = parts['head']
        pivot_head = (head_crop.shape[1] // 2, head_crop.shape[0] - 10)
        rotated_head = rotate_image(head_crop, angle_head, pivot_head)
        canvas = paste_part(canvas, rotated_head, head_pos)

        # Helper to rotate limb around first keypoint
        def rotate_and_paste(limb_name, angle, keypoint_idx):
            crop, pos = parts[limb_name]
            pivot = (keypoints[keypoint_idx][0] - pos[0], keypoints[keypoint_idx][1] - pos[1])
            rotated = rotate_image(crop, angle, pivot)
            nonlocal canvas
            canvas = paste_part(canvas, rotated, pos)

        # Left arm
        rotate_and_paste('left_upper_arm', angle_upper_arm, 11)
        rotate_and_paste('left_lower_arm', angle_lower_arm, 13)

        # Right arm
        rotate_and_paste('right_upper_arm', -angle_upper_arm, 12)  # opposite swing
        rotate_and_paste('right_lower_arm', -angle_lower_arm, 14)

        # Left leg
        rotate_and_paste('left_upper_leg', -angle_upper_leg, 23)
        rotate_and_paste('left_lower_leg', -angle_lower_leg, 25)

        # Right leg
        rotate_and_paste('right_upper_leg', angle_upper_leg, 24)
        rotate_and_paste('right_lower_leg', angle_lower_leg, 26)

        frames.append(canvas)

    return frames
