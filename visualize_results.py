import cv2
import json
import numpy as np

def visualize_first_frame():
    json_path = "outputs/Karate_vrg.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    video_path = "Karate.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video")
        return

    # Get tracks for frame 1
    tracks = [t for t in data['tracks'] if t['frame'] == 1]
    
    for t in tracks:
        bbox = t['bbox']['refined']
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        for name, pt in t['keypoints'].items():
            x, y = int(pt['x']), int(pt['y'])
            # Draw point if within some reasonable bounds for visualization
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite("outputs/debug_frame_1.jpg", frame)
    print("Saved outputs/debug_frame_1.jpg")

if __name__ == "__main__":
    visualize_first_frame()
