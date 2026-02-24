import cv2
import numpy as np

def create_test_video(output_path="test.mp4", width=640, height=480, fps=30, duration=2):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a moving rectangle as a "person" placeholder
        x = (i * 5) % width
        y = height // 2
        cv2.rectangle(frame, (x, y), (x + 50, y + 100), (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")

if __name__ == "__main__":
    create_test_video()
