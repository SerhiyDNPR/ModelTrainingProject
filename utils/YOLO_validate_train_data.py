# import os
# import cv2
# from ultralytics import YOLO
# from glob import glob
# import random
# import tkinter as tk
# from tkinter import filedialog

# def load_ground_truth_boxes(label_path, img_width, img_height):
#     """Завантажує ground truth бокси з YOLO .txt файлу та конвертує їх у піксельні координати."""
#     gt_boxes = []
#     if not os.path.exists(label_path):
#         return gt_boxes
        
#     with open(label_path, 'r') as f:
#         for line in f.readlines():
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
            
#             _, x_c, y_c, w, h = map(float, parts)
            
#             # Конвертація з нормалізованих координат у пікселі
#             x_min = int((x_c - w / 2) * img_width)
#             y_min = int((y_c - h / 2) * img_height)
#             x_max = int((x_c + w / 2) * img_width)
#             y_max = int((y_c + h / 2) * img_height)
            
#             gt_boxes.append((x_min, y_min, x_max, y_max))
            
#     return gt_boxes

# def draw_legend(image):
#     """Малює легенду на зображенні."""
#     legend_items = [
#         ("Prediction", (255, 0, 0)),    # Синій
#         ("Ground Truth", (0, 0, 255)) # Червоний
#     ]
#     start_y = 30
    
#     for text, color in legend_items:
#         cv2.putText(image, text, (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#         start_y += 40

# def main():
#     # --- ВЗАЄМОДІЯ З КОРИСТУВАЧЕМ ---
#     root = tk.Tk()
#     root.withdraw() # Ховаємо головне вікно tkinter

#     root.attributes('-topmost', True)

#     # 1. Вибір файлу моделі
#     model_path = filedialog.askopenfilename(
#         title="Оберіть файл моделі (.pt)",
#         filetypes=[("PyTorch Models", "*.pt")]
#     )

#     root.attributes('-topmost', False)

#     if not model_path:
#         print("Модель не обрано. Роботу завершено.")
#         return

#     # 2. Вибір папки з датасетом
#     dataset_dir = filedialog.askdirectory(
#         title="Оберіть кореневу папку датасету (напр., YoloDataset)"
#     )
#     if not dataset_dir:
#         print("Папку з датасетом не обрано. Роботу завершено.")
#         return

#     CONF_THRESHOLD = 0.25
#     # ------------------------------------

#     if not os.path.exists(model_path):
#         print(f"Помилка: Файл моделі не знайдено за шляхом '{model_path}'")
#         return
        
#     print(f"Завантаження моделі з '{model_path}'...")
#     model = YOLO(model_path)

#     image_paths = glob(os.path.join(dataset_dir, 'images', '**', '*.png'), recursive=True)
#     if not image_paths:
#         print(f"Помилка: Зображення не знайдено у '{os.path.join(dataset_dir, 'images')}'")
#         return

#     random.shuffle(image_paths)

#     print(f"Знайдено {len(image_paths)} зображень. Починаємо візуалізацію.")
#     print("Натисніть 'q' або 'Esc' для виходу, або будь-яку іншу клавішу для наступного зображення.")

#     window_name = 'Ground Truth vs Prediction'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#     for image_path in image_paths:
#         print(f"\nОбробка зображення: {image_path}")
        
#         image = cv2.imread(image_path)
#         if image is None:
#             print(" -> Не вдалося завантажити зображення.")
#             continue
#         h, w, _ = image.shape

#         results = model(image, conf=CONF_THRESHOLD, verbose=False)
#         image_to_display = image.copy()

#         # Малюємо розмітку з датасету (Ground Truth)
#         # **ВИПРАВЛЕНО**: Більш надійний спосіб знайти відповідний файл мітки
#         images_base_dir = os.path.join(dataset_dir, 'images')
#         relative_image_path = os.path.relpath(image_path, images_base_dir)
#         relative_label_path = os.path.splitext(relative_image_path)[0] + ".txt"
#         label_path = os.path.join(dataset_dir, 'labels', relative_label_path)
        
#         print(f" -> Шукаю мітку: {label_path}") # Діагностичне повідомлення
        
#         gt_boxes = load_ground_truth_boxes(label_path, w, h)
#         if not gt_boxes:
#             print(" -> Мітку не знайдено або вона порожня.") # Діагностичне повідомлення
            
#         for (x1, y1, x2, y2) in gt_boxes:
#             cv2.rectangle(image_to_display, (x1, y1), (x2, y2), (0, 0, 255), 2) # Червоний
#             cv2.putText(image_to_display, 'Ground Truth', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Малюємо результати розпізнавання (Prediction)
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf, cls_id = box.conf[0], int(box.cls[0])
#             class_name = model.names[cls_id]
#             label = f"{class_name} {conf:.2f}"
            
#             cv2.rectangle(image_to_display, (x1, y1), (x2, y2), (255, 0, 0), 2) # Синій
#             cv2.putText(image_to_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#         # Додаємо легенду на зображення
#         draw_legend(image_to_display)
        
#         cv2.imshow(window_name, image_to_display)
        
#         key = cv2.waitKey(0)
#         if key == ord('q') or key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
#             break

#     cv2.destroyAllWindows()
#     print("\nРоботу завершено.")

# if __name__ == '__main__':
#     main()