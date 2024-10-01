import os
import cv2

def load_video_frames(video_dir):
    frame_names = [
        f for f in os.listdir(video_dir) if f.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'))
    ]
    frame_names.sort(key=lambda f: int(os.path.splitext(f)[0]))
    return frame_names

def load_frame(video_dir, frame_name):
    return cv2.imread(os.path.join(video_dir, frame_name))

def navigate_frame(current_idx, direction, total_frames):
    if direction == "left" and current_idx > 0:
        return current_idx - 1
    elif direction == "right" and current_idx < total_frames - 1:
        return current_idx + 1
    return current_idx