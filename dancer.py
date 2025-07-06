from beat_detector import get_beats
from image_animator import load_image, generate_frames
from pose_estimator import get_pose_landmarks
import moviepy.editor as mpy
import os

IMG_PATH = "assets/input.jpg"
AUDIO_PATH = "assets/song.mp3"
OUTPUT_PATH = "output/dance_video.mp4"

def main():
    print("🔍 Detecting beats...")
    beat_times = get_beats(AUDIO_PATH)

    print("🧍 Getting pose landmarks...")
    result = get_pose_landmarks(IMG_PATH)
    if not result:
        print("❌ No pose detected, exiting.")
        return
    keypoints, _ = result

    print("🖼️ Loading image...")
    img = load_image(IMG_PATH)

    print("🎞️ Generating frames...")
    frames = generate_frames(img, keypoints, beat_times, duration=10)
    print(f"✅ Generated {len(frames)} frames")

    try:
        print("📤 Exporting video...")
        clip = mpy.ImageSequenceClip(frames, fps=30)
        clip = clip.set_audio(mpy.AudioFileClip(AUDIO_PATH).subclip(0, 10))
        os.makedirs("output", exist_ok=True)
        clip.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac")
        print(f"✅ Video saved to {OUTPUT_PATH}")
    except Exception as e:
        print("❌ Video export failed:", e)

if __name__ == "__main__":
    main()
