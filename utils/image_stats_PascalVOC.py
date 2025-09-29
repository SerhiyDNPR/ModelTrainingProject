import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.ticker as mticker

def select_directory():
    """Відкриває діалогове вікно для вибору директорії."""
    root = tk.Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory(title="Оберіть папку з файлами розмітки Pascal VOC (.xml)")
    return directory_path

def parse_pascal_voc_files(directory_path):
    """
    Проходиться по всіх .xml файлах у директорії, парсить їх
    і повертає списки з розмірами зображень та об'єктів.
    """
    xml_files = glob(os.path.join(directory_path, '*.xml'))
    
    if not xml_files:
        print(f"❌ У папці '{directory_path}' не знайдено файлів .xml.")
        return None

    print(f"🔎 Знайдено {len(xml_files)} файлів для аналізу. Обробка...")

    image_widths, image_heights = [], []
    bbox_widths, bbox_heights = [], []

    for xml_file in tqdm(xml_files, desc="Аналіз файлів"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            size_element = root.find('size')
            if size_element:
                img_width = int(size_element.find('width').text)
                img_height = int(size_element.find('height').text)
                image_widths.append(img_width)
                image_heights.append(img_height)

            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                if bndbox:
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    
                    bbox_w = xmax - xmin
                    bbox_h = ymax - ymin
                    
                    if bbox_w > 0 and bbox_h > 0:
                        bbox_widths.append(bbox_w)
                        bbox_heights.append(bbox_h)
        except Exception as e:
            print(f"\n⚠️ Помилка при обробці файлу {os.path.basename(xml_file)}: {e}")

    return {
        "image_widths": image_widths,
        "image_heights": image_heights,
        "bbox_widths": bbox_widths,
        "bbox_heights": bbox_heights
    }

def plot_distributions(stats):
    """
    Будує та відображає гістограми розподілу на основі зібраної статистики.
    """
    if not stats or (not stats["image_widths"] and not stats["bbox_widths"]):
        print("Немає даних для побудови графіків.")
        return

    # --- Графік 1: Розміри зображень (залишаємо в лінійній шкалі) ---
    if stats["image_widths"]:
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig1.suptitle('Розподіл розмірів зображень (Лінійна шкала)', fontsize=16)

        ax1.hist(stats["image_widths"], bins=50, color='skyblue', edgecolor='black')
        ax1.set_title('Розподіл ширин зображень')
        ax1.set_xlabel('Ширина (пікселі)')
        ax1.set_ylabel('Кількість')
        
        ax2.hist(stats["image_heights"], bins=50, color='salmon', edgecolor='black')
        ax2.set_title('Розподіл висот зображень')
        ax2.set_xlabel('Висота (пікселі)')
        ax2.set_ylabel('Кількість')
        
        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Графік 2: Розміри об'єктів (bounding boxes)
    if stats["bbox_widths"]:
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
        fig2.suptitle('Розподіл розмірів об\'єктів (Логарифмічна шкала)', fontsize=16)
        
        # --- Графік для ширин об'єктів (ax3) ---
        bbox_widths = [w for w in stats["bbox_widths"] if w > 0]
        if bbox_widths:
            min_w = np.floor(np.log10(min(bbox_widths)))
            max_w = np.ceil(np.log10(max(bbox_widths)))
            bins_w = np.logspace(min_w, max_w, num=75)

            ax3.hist(bbox_widths, bins=bins_w, color='lightgreen', edgecolor='black')
            ax3.set_xscale('log')
            ax3.set_title('Розподіл ширин об\'єктів')
            ax3.set_xlabel('Ширина (пікселі)')
            ax3.set_ylabel('Кількість')
            ax3.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax3.xaxis.get_major_formatter().set_scientific(False)
            ax3.minorticks_off()
            ax3.grid(True, which="both", linestyle='--', alpha=0.6)

        # --- Графік для висот об'єктів (ax4) ---
        bbox_heights = [h for h in stats["bbox_heights"] if h > 0]
        if bbox_heights:
            min_h = np.floor(np.log10(min(bbox_heights)))
            max_h = np.ceil(np.log10(max(bbox_heights)))
            bins_h = np.logspace(min_h, max_h, num=75)

            ax4.hist(bbox_heights, bins=bins_h, color='gold', edgecolor='black')
            ax4.set_xscale('log')
            ax4.set_title('Розподіл висот об\'єктів')
            ax4.set_xlabel('Висота (пікселі)')
            ax4.set_ylabel('Кількість')
            ax4.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax4.xaxis.get_major_formatter().set_scientific(False)
            ax4.minorticks_off()
            ax4.grid(True, which="both", linestyle='--', alpha=0.6)
        
        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    print("\n✅ Аналіз завершено. Відображення графіків...")
    plt.show()

if __name__ == "__main__":
    data_dir = select_directory()
    
    if data_dir:
        statistics = parse_pascal_voc_files(data_dir)
        
        if statistics:
            plot_distributions(statistics)
    else:
        print("Директорію не обрано. Завершення роботи.")