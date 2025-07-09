from beat_detector import get_beats, load_image
from pose_estimator import get_pose_landmarks_3d
from image_animator import generate_frames
import moviepy.editor as mpy

IMG_PATH = "assets/input.jpg"
AUDIO_PATH = "assets/song.mp3"
OUTPUT_PATH = "output/dance_video.mp4"

def main():
    print("Detecting beats...")
    beat_times = get_beats(AUDIO_PATH)

    print("Getting 3D pose landmarks...")
    keypoints_2d, keypoints_3d = get_pose_landmarks_3d(IMG_PATH)
    if not keypoints_2d:
        print("No pose detected, exiting.")
        return

    print("Loading image...")
    img = load_image(IMG_PATH)

    print("Generating frames...")
    frames = generate_frames(img, keypoints_2d, keypoints_3d, beat_times, duration=10)

    print("Exporting video...")
    clip = mpy.ImageSequenceClip(frames, fps=30)
    clip = clip.set_audio(mpy.AudioFileClip(AUDIO_PATH).subclip(0, 10))
    clip.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    main()
