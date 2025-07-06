import librosa
from image_animator import generate_frames
from pose_estimator import get_pose_landmarks
from PIL import Image
import numpy as np

def get_beats(audio_path):
    y, sr = librosa.load(audio_path)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return beat_times

def load_image(image_path):
    img = Image.open(image_path).convert("RGBA")
    return np.array(img)
