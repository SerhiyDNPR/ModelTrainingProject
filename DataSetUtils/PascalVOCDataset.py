import os
import glob
import torch
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image

class PascalVOCDataset(torch.utils.data.Dataset):
    """
    Кастомний Dataset для PASCAL VOC, що кешує дані в RAM для швидкого доступу.
    """
    def __init__(self, root_dir, transforms, label_map):
        self.root_dir = root_dir
        self.transforms = transforms
        self.label_map = label_map
        self.xml_files = sorted(glob.glob(os.path.join(root_dir, "*.xml")))
        
        # --- Кешування даних ---
        self.data_cache = []
        print(f"\n⏳ Кешування даних з '{root_dir}' в оперативну пам'ять...")
        
        # Фільтруємо фонові зображення, якщо це тренувальна вибірка
        files_to_process = []
        if 'train' in root_dir:
            for xml_path in self.xml_files:
                tree = ET.parse(xml_path)
                if tree.getroot().find('object') is not None:
                    files_to_process.append(xml_path)
            if len(files_to_process) < len(self.xml_files):
                print(f"🔍 Для '{os.path.basename(root_dir)}' відфільтровано {len(self.xml_files) - len(files_to_process)} фонових зображень.")
        else:
            files_to_process = self.xml_files

        for xml_path in tqdm(files_to_process, desc="Завантаження даних"):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            img_filename = root.find('filename').text
            img_path = os.path.join(self.root_dir, img_filename)
            img = Image.open(img_path).convert("RGB")

            boxes = []
            labels = []
            for member in root.findall('object'):
                class_name = member.find('name').text
                if class_name in self.label_map and class_name != '__background__':
                    bndbox = member.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.label_map[class_name])
            
            # Додаємо в кеш лише валідні дані
            self.data_cache.append({
                "image": img,
                "boxes": boxes,
                "labels": labels
            })
        print("✅ Кешування завершено.")

    def __len__(self):
        # Довжина тепер дорівнює розміру кешу
        return len(self.data_cache)
    
    def __iter__(self):
            for i in range(len(self)):
                yield self[i]

    def __getitem__(self, idx):
        # start_time = time.time() # Для діагностики
        
        cached_data = self.data_cache[idx]
        
        img = cached_data["image"].copy() # Копіюємо, щоб трансформації не псували кеш
        boxes = torch.as_tensor(cached_data["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(cached_data["labels"], dtype=torch.int64)
        
        # Перевірка на випадок, якщо у val/test потрапив файл без об'єктів
        if boxes.shape[0] == 0:
            # Створюємо "пустий" target, але з правильними типами
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([idx])
            }
        else:
            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([idx])
            }
        
        # print(f"__getitem__ took {time.time() - start_time:.6f} seconds") # Для діагностики
        return img, target