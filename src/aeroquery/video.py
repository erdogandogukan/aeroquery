import cv2
from aeroquery.detect import get_detector
from aeroquery.storage import save_detection
from aeroquery.rag import create_report_from_detections, add_report

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

            try:
                detections = detector.predict(frame)
                for d in detections:
                    save_detection(d.class_name, d.confidence, d.bbox)

                report = create_report_from_detections(detections)
                add_report(report)

                print(f"Frame {frame_count}: {len(detections)} detect")

            except Exception as e:
                print(f"Frame {frame_count}  failed, skipping: {e}")   
            
        frame_count += 1

    cap.release()

    print(f"\nTotal {frame_count} frames read, {process_count} frames processed")


def process_video_to_output(video_path: str, output_path: str = "output.mp4"):
    cap = cv2.VideoCapture(video_path)
    detector = get_detector()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        detections = detector.predict(frame)

        for d in detections:
            x1, y1, x2, y2 = d.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, d.class_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"{frame_count} frames read, '{output_path}' was created.")      


def process_webcam(frame_skip: int = 5):
    
    cap = cv2.VideoCapture(0)

    detector = get_detector()
    frame_count = 0
    last_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        if frame_count % frame_skip == 0:
            last_detections = detector.predict(frame)

        
        for d in last_detections:
            x1, y1, x2, y2 = d.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, d.class_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("AeroQuery Live", frame)

        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")    