"""
Розпізнавання об'єктів на відео в реальному часі з виділенням рамкою
Використовуємо веб-камеру, YOLOv5 для детекції та ResNet50 для класифікації
"""

import torch # бібліотека для роботи з нейронними мережами
import torchvision.transforms as transforms # трансформації зображень
import torchvision.models as models # попередньо навчені моделі
import cv2 # бібліотека для обробки зображень та відео
import requests # для завантаження файлів з інтернету
import numpy as np # для роботи з масивами
import time # для вимірювання часу

# Завантаження моделі YOLOv5 для детекції об'єктів
print("Завантаження моделі YOLOv5...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval() # встановлення моделі в режим оцінки

# Завантаження моделі ResNet50 для класифікації об'єктів
print("Завантаження моделі ResNet50...")
model = models.resnet50(pretrained=True)
model.eval() # встановлення моделі в режим оцінки

# Визначаємо GPU, якщо доступна
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
yolo_model = yolo_model.to(device)

# Завантаження міток класів ImageNet
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Трансформації кадрів для відео
transform = transforms.Compose([
    transforms.ToPILImage(), # конвертація в PIL Image (зручний формат для трансформацій)
    transforms.Resize(256), # зміна розміру до 256x256
    transforms.CenterCrop(224), # центральне обрізання до 224x224
    transforms.ToTensor(), # конвертація в тензор
    transforms.Normalize( # нормалізація зображення
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_frame(frame):
    """Розпізнання об'єктів на кадрі відео"""
    # Конвертація кадру з BGR в RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Застуємо трансформації
    img_tensor = transform(frame_rgb).unsqueeze(0).to(device) # додавання батч розміру та переміщення на пристрій

    # Робимо передбачення з ResNet50
    with torch.no_grad():
        outputs = model(img_tensor)

    # Отримує ймовірності
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Отримуємо найкраще передбачення
    top_prob, top_idx = torch.max(probabilities, 0)

    top_class = labels[top_idx.item()]
    confidence = top_prob.item() * 100 # точність у відсотках

    return top_class, confidence

def run_realtime_detection(camera_id=0, confidence_threshold=0.3):
    # Відкриваємо відеопотік з веб-камери
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Не вдалося відкрити камеру.")
        return

    #  Встановлюємо розмір кадру
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Початок розпізнавання об'єктів. Натисніть 'q' для виходу.")

    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    # Кольори для різних класів об'єктів
    colors = [
        (0, 255, 0),  # Зелений
        (255, 0, 0),  # Синій
        (0, 0, 255),  # Червоний
        (255, 255, 0),  # Блакитний
        (255, 0, 255),  # Пурпурний
        (0, 255, 255),  # Жовтий
    ]

    while True:
        ret, frame = cap.read() # зчитування кадру
        if not ret:
            print("Не вдалося отримати кадр з камери.")
            break

        # Детекція зображення за допомогою YOLOv5
        try:
            results = yolo_model(frame)
            detections = results.panads().xyxy[0] # отримання детекцій

            # Обробка кожної детекції
            for idx, detection in detections.iterrows():
                confidence = detection["confidence"]
                if confidence < confidence_threshold:
                    continue

                # Координати координати рамки
                x1, y1, x2, y2 = int(detection["xmin"]), int(detection["ymin"]), int(detection["xmax"]), int(detection["ymax"])

                # Отримуємо найменування класу
                class_name = detection["name"]

                # Вибираємо колір для рамки
                calor = colors[idx % len(colors)]

                # Малюємо рамку навколо об'єкта
                cv2.rectangle(frame, (x1, y1), (x2, y2), calor, 2)

                # Підготовуємо текст для відображення
                label = f"{class_name} -- {confidence:.2f}"

                # Малюємо фон для тексту
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), calor, -1)

                # Малюємо текст
                cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        except Exception  as e:
            print(f"Помилка детекції: {e}")
            continue

        fps_counter += 1
        if (time.time() - fps_start_time) >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        # Відображення FPS на кадрі
        fps_text = f"FPS: {fps}"
        (text_width, text_height), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(frame, (5, frame.shape[0] - 30),
                      (15 + text_width, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Відображаємо інструкції
        cv2.putText(frame, "Natysni 'q' - vyhid",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        # Відображення кадру
        cv2.imshow('Real-Time Object Detection', frame)

        # Вихід з циклу при натисканні 'q'
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Вихід з програми.")
            break

    # Звільнення ресурсів
    cap.release()
    cv2.destroyAllWindows()
    print("Програма завершила роботу.")



def main():
   print("Hello, World!")


if __name__ == '__main__':
    main()
