from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


class Detector:
    def __init__(self, weights_path: str) -> None:
        self.model = YOLO(weights_path)
        
    def predict(self, image_path: str) -> list[Detection]:
        results = self.model(image_path)
        r = results[0]


        detections = []

        for box in r.boxes:
            cls_index = int(box.cls.item())
            class_name = r.names[cls_index]
            confidence = box.conf.item()
            bbox = tuple(box.xyxy.tolist()[0])

            detections.append(Detection(class_name, confidence, bbox))
        
        return detections