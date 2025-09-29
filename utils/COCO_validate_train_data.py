import os
import cv2
import json
import tkinter as tk
from tkinter import filedialog

class COCODatasetValidator:
    """
    Клас для візуальної перевірки розмітки датасету у форматі COCO.
    Дозволяє переглядати зображення з нанесеними рамками та класами.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'train')
        self.annotation_path = self.find_annotation_file()

        if not self.annotation_path:
            raise FileNotFoundError("Не знайдено JSON-файл з анотаціями в папці 'annotations'.")

        print(f"Завантаження анотацій з: {self.annotation_path}")
        with open(self.annotation_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # --- Створення зручних структур даних ---
        self.images = self.coco_data['images']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
            
        print(f"Знайдено {len(self.images)} зображень та {len(self.coco_data['annotations'])} анотацій.")
        print(f"Класи: {list(self.categories.values())}")

    def find_annotation_file(self):
        """Знаходить основний файл анотацій у підпапці 'annotations'."""
        ann_dir = os.path.join(self.root_dir, 'annotations')
        if not os.path.isdir(ann_dir):
            return None
        
        for fname in os.listdir(ann_dir):
            if fname.startswith('instances') and fname.endswith('.json'):
                return os.path.join(ann_dir, fname)
        return None

    def draw_annotations(self, image, image_info, annotations_for_image):
        """Малює рамки та назви класів на зображенні."""
        for ann in annotations_for_image:
            bbox = ann['bbox']
            category_id = ann['category_id']
            class_name = self.categories.get(category_id, "Unknown")
            
            # COCO bbox format is [x_min, y_min, width, height]
            x, y, w, h = map(int, bbox)
            
            # Малюємо прямокутник
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Готуємо текст
            label = f"{class_name}"
            
            # Малюємо фон для тексту та сам текст
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x, y - text_height - 10), (x + text_width, y), (0, 255, 0), -1)
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Додаємо інформацію про файл
        img_h, img_w, _ = image.shape
        info_text = f"{image_info['file_name']} [{img_w}x{img_h}]"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


    def run(self):
        """Запускає цикл перегляду зображень."""
        cv2.namedWindow('COCO Validator', cv2.WINDOW_NORMAL)
        current_index = 0
        
        while True:
            image_info = self.images[current_index]
            image_path = os.path.join(self.image_dir, image_info['file_name'])
            
            if not os.path.exists(image_path):
                print(f"Попередження: Файл зображення не знайдено: {image_path}")
                # Створюємо чорне зображення-заглушку, щоб не переривати процес
                image = cv2.UMat(np.zeros((image_info['height'], image_info['width'], 3), dtype=np.uint8))
            else:
                image = cv2.imread(image_path)
            
            # Отримуємо анотації для поточного зображення
            image_id = image_info['id']
            annotations_for_image = self.annotations.get(image_id, [])
            
            # Малюємо анотації
            self.draw_annotations(image, image_info, annotations_for_image)
            
            # Показуємо прогрес
            progress_text = f"Image: {current_index + 1} / {len(self.images)}"
            cv2.putText(image, progress_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('COCO Validator', image)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('d') or key == ord(' '):  # 'd' -> наступне зображення
                current_index = (current_index + 1) % len(self.images)
            elif key == ord('a'):  # 'a' -> попереднє зображення
                current_index = (current_index - 1 + len(self.images)) % len(self.images)
            elif key == ord('q') or key == 27:  # 'q' або ESC -> вихід
                break
                
        cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.withdraw()
    
    print("Будь ласка, оберіть кореневу директорію вашого датасету у форматі COCO...")
    dataset_path = filedialog.askdirectory(title="Оберіть папку датасету COCO")
    
    if not dataset_path:
        print("Папку не обрано. Вихід.")
        return
        
    try:
        validator = COCODatasetValidator(dataset_path)
        print("\n--- Навігація ---")
        print(" 'd' або ' '-> Наступне зображення")
        print(" 'a' -> Попереднє зображення")
        print(" 'q' або ESC -> Вихід")
        print("-----------------\n")
        validator.run()
    except Exception as e:
        print(f"\nСталася помилка: {e}")
        print("Переконайтеся, що ви обрали кореневу папку датасету COCO, яка містить підпапки 'annotations' та 'train'.")

if __name__ == '__main__':
    main()