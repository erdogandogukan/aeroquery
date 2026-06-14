from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


class Detector:
    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        # TODO: modeli bir kez yükle ve self.model'e ata
        self.model = YOLO(weights_path)

    def predict(self, image_path: str) -> list[Detection]:
        # 1) modeli görüntü üzerinde çalıştır
        results = self.model(image_path)
        r = results[0]

        # 2) sonuçları toplayacağın boş bir liste
        detections = []

        # 3) her tespit için döngü — kaç tane var? len(r.boxes) kadar
        for i in range(len(r.boxes)):
            # TODO: sınıf indeksini al (cls[i]), int'e çevir
            cls_index = int(r.boxes[i].cls)
            # TODO: r.names sözlüğünden ismi bul
            class_name = r.names[cls_index]
            # TODO: güveni al, .item() ile float yap
            confidence = r.boxes[i].conf.item()
            # TODO: kutuyu al (xyxy[i]), .tolist() ile listeye çevir
            box = r.boxes[i].xyxy.tolist()

            # TODO: bir Detection oluştur ve detections'a ekle
            detection = Detection(class_name=class_name, confidence=confidence, bbox=tuple(box))
            detections.append(detection)

        return detections