from ultralytics import YOLO
import cv2
import os
from pathlib import Path


def detect_fish():

    current_dir = Path(__file__).parent.absolute()
    model_path = current_dir / 'runs' / 'detect' / 'fish_detector' / 'weights' / 'best.pt'
    model = YOLO(model_path)
    valid_path = current_dir / 'data' / 'valid' / 'images'

    valid_images = list(valid_path.glob('*.jpg')) + list(valid_path.glob('*.png'))

    for img_path in valid_images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"błąd {img_path}")
            continue

        results = model(img)[0]

        boxes = results.boxes
        num_fish = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"rybisko: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(img, f"Liczebnosc lawicy: {num_fish}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        title = f"Liczydlo lawicy rybek - {img_path.name}"
        cv2.imshow(title, img)

        key = cv2.waitKey(0)

        if key in [13, 32]:
            cv2.destroyAllWindows()
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()



if __name__ == '__main__':
    detect_fish()
