import cv2
import os
from PIL import Image

def extract_frames(video_path, output_dir, subsample_rate=1):
    """
    Extracts frames from a video file and saves them as RGB images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the extracted images will be saved.
        subsample_rate (int): Save every nth frame (default is 1, which saves all frames).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at: {output_dir}")

    # Open the video file
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 0
    saved_count = 0

    print("Starting frame extraction...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished reading video file.")
            break

        if frame_count % subsample_rate == 0:
            # Convert the frame from BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Save the frame as an image using PIL
            output_path = os.path.join(output_dir, f"{saved_count:04d}.png")
            image = Image.fromarray(frame_rgb)
            image.save(output_path)
            saved_count += 1

            if saved_count % 100 == 0:
                print(f"{saved_count} frames saved...")

        frame_count += 1

    cap.release()
    print(f"Extraction completed. Saved {saved_count} frames out of {frame_count} total frames.")

if __name__ == "__main__":
    # Parameters
    video_file = 'rosbags/IMG_1239.MOV'              # Path to your .MOV video file
    output_folder = 'results/calib/rgbs'             # Output directory for extracted frames
    subsample_rate = 5                               # Change this to save every nth frame

    extract_frames(video_file, output_folder, subsample_rate)