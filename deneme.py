from aeroquery.detect import Detector

detector = Detector(r"models/best.pt")
sonuclar = detector.predict(r"models\0000074_09738_d_0000019.jpg")

print(f"{len(sonuclar)} tespit bulundu\n")
for d in sonuclar[:5]:
    print(d)