import cv2
from aeroquery.detect import get_detector
from aeroquery.storage import save_detection

def process_video(video_path: str, frame_skip: int = 30):
    
    cap = cv2.VideoCapture(video_path)
    detector = get_detector()

    frame_count = 0
    process_count = 0

    while True:
        ret,frame = cap.read()

        if not ret:
            break
        if frame_count % frame_skip == 0:
            process_count += 1

            detections = detector.predict(frame)
            for d in detections:
                save_detection(d.class_name, d.confidence, d.bbox)

            print(f"Kare {frame_count}: {len(detections)} tespit")
            
        frame_count += 1

    cap.release()

    print(f"\nToplam {frame_count} kare okundu, {process_count} kare islendi")