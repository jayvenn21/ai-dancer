import numpy as np
import cv2

BODY_PARTS = {
    'head': [0, 1, 2, 3, 7],
    'left_upper_arm': [11, 13],
    'left_lower_arm': [13, 15],
    'right_upper_arm': [12, 14],
    'right_lower_arm': [14, 16],
    'left_upper_leg': [23, 25],
    'left_lower_leg': [25, 27],
    'right_upper_leg': [24, 26],
    'right_lower_leg': [26, 28],
    'torso': [11, 12, 23, 24]
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

def get_perspective_transform(src_pts, dst_pts):
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return matrix

def warp_part(part_img, matrix, dst_shape):
    warped = cv2.warpPerspective(part_img, matrix, (dst_shape[1], dst_shape[0]), borderMode=cv2.BORDER_TRANSPARENT)
    return warped

def generate_frames(image_np, keypoints_2d, keypoints_3d, beat_times, fps=30, duration=10):
    frames = []
    total_frames = int(fps * duration)
    beat_frames = set(int(bt * fps) for bt in beat_times)

    parts = {}
    for part_name, idxs in BODY_PARTS.items():
        pts_2d = [keypoints_2d[i] for i in idxs]
        xs, ys = zip(*pts_2d)
        crop, pos = crop_part(image_np, (min(xs), min(ys)), (max(xs), max(ys)), padding=30)
        parts[part_name] = {
            'crop': crop,
            'pos': pos,
            'original_pts': pts_2d,
        }

    for frame_i in range(total_frames):
        canvas = np.zeros_like(image_np)

        torso_angle = 10 * np.sin(2 * np.pi * (frame_i / total_frames))
        on_beat = frame_i in beat_frames

        for part_name, data in parts.items():
            crop = data['crop']
            pos_x, pos_y = data['pos']
            orig_pts_2d = data['original_pts']
            part_pts_3d = [keypoints_3d[i] for i in BODY_PARTS[part_name]]

            angle_deg = 0
            if part_name == 'torso':
                angle_deg = torso_angle
            elif 'arm' in part_name:
                angle_deg = 20 if on_beat else 0
                if 'right' in part_name:
                    angle_deg = -angle_deg
            elif 'leg' in part_name:
                angle_deg = 10 * np.sin(2 * np.pi * (frame_i / total_frames))
            elif part_name == 'head':
                angle_deg = 5 * np.sin(2 * np.pi * (frame_i / total_frames))

            angle_rad = np.deg2rad(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            cx, cy, cz = keypoints_3d[11]

            rotated_pts_3d = []
            for x, y, z in part_pts_3d:
                dx, dy, dz = x - cx, y - cy, z - cz
                rx = cos_a * dx + sin_a * dz
                rz = -sin_a * dx + cos_a * dz
                rotated_pts_3d.append((cx + rx, cy + dy, cz + rz))

            dst_pts = [(int(x), int(y)) for x, y, z in rotated_pts_3d]

            h, w = crop.shape[:2]
            src_pts = [(0, 0), (w, 0), (w, h), (0, h)]
            if len(dst_pts) >= 4:
                dst_quad = [dst_pts[0], dst_pts[-1], dst_pts[-2], dst_pts[1]]
            else:
                dst_quad = src_pts

            dst_quad_offset = [(x - pos_x, y - pos_y) for x, y in dst_quad]

            matrix = get_perspective_transform(src_pts, dst_quad_offset)
            warped_part = warp_part(crop, matrix, (h, w))

            y1, y2 = pos_y, pos_y + h
            x1, x2 = pos_x, pos_x + w

            if y1 < 0 or x1 < 0 or y2 > canvas.shape[0] or x2 > canvas.shape[1]:
                continue

            alpha = warped_part[:, :, 3] / 255.0 if warped_part.shape[2] == 4 else None
            if alpha is not None:
                for c in range(3):
                    canvas[y1:y2, x1:x2, c] = (
                        alpha * warped_part[:, :, c] +
                        (1 - alpha) * canvas[y1:y2, x1:x2, c]
                    )
            else:
                canvas[y1:y2, x1:x2] = warped_part

        frames.append(canvas)

    return frames
