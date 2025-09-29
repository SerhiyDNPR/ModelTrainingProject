import os
import glob
import json
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import CheckButtons, RectangleSelector
import matplotlib.ticker as mticker
import numpy as np

class ImageViewer:
    """
    Клас для відображення зображень в окремому вікні з можливістю
    навігації за допомогою стрілок та закриття по клавіші ESC.
    Тепер також відображає розширену інформацію.
    """
    def __init__(self, image_objects, selected_range, dimension_type):
        if not image_objects:
            print("⚠️ Не знайдено об'єктів для перегляду в заданому діапазоні.")
            return

        self.image_objects = image_objects
        self.selected_range = selected_range
        self.dimension_type = dimension_type # 'widths' or 'heights'
        self.current_index = 0

        self.fig, self.ax = plt.subplots()
        
        if self.fig.canvas.manager:
            self.fig.canvas.manager.set_window_title("Перегляд зображень")
        
        self.fig.instance = self
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.show_image()
        
        try:
            self.fig.canvas.get_tk_widget().focus_force()
        except AttributeError:
            print("\nПорада: Клацніть на вікні з зображенням, щоб активувати керування клавішами.")

        plt.show(block=True)

    def show_image(self):
        """Відображає поточне зображення та розширену інформацію."""
        self.ax.clear()
        
        # Отримуємо дані поточного об'єкта
        path, width, height = self.image_objects[self.current_index]
        
        try:
            image = mpimg.imread(path)
            self.ax.imshow(image)
        except FileNotFoundError:
            self.ax.text(0.5, 0.5, f"Файл не знайдено:\n{path}", ha='center', va='center', color='red')
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Помилка завантаження:\n{os.path.basename(path)}\n{e}", ha='center', va='center', color='red')

        # --- НОВА ЛОГІКА: ФОРМУВАННЯ РОЗШИРЕНОГО ЗАГОЛОВКА ---
        range_min, range_max = self.selected_range
        
        # Форматуємо інформацію про розміри
        width_str = f"W: {width}"
        height_str = f"H: {height}"
        
        # Виділяємо жирним той розмір, за яким відбувався відбір
        if self.dimension_type == 'widths':
            info_str = f"**{width_str}**, {height_str}"
        else:
            info_str = f"{width_str}, **{height_str}**"

        title_line1 = f"Об'єкт: {self.current_index + 1}/{len(self.image_objects)} | Діапазон: [{range_min:.0f}-{range_max:.0f}] px"
        title_line2 = f"Розміри об'єкта: {info_str.replace('**', '')}" # Відображення без зірочок
        title_line3 = os.path.basename(path)
        
        full_title = f"{title_line1}\n{title_line2}\n{title_line3}"
        
        # Для візуального виділення використовуємо text замість title
        self.ax.set_title(full_title)
        
        self.ax.axis('off')
        self.fig.canvas.draw()

    def on_key_press(self, event):
        """Обробляє натискання клавіш для навігації та виходу."""
        if event.key == 'right':
            self.current_index = (self.current_index + 1) % len(self.image_objects)
            self.show_image()
        elif event.key == 'left':
            self.current_index = (self.current_index - 1 + len(self.image_objects)) % len(self.image_objects)
            self.show_image()
        elif event.key == 'escape':
            plt.close(self.fig)

def select_root_directory():
    """Відкриває діалогове вікно для вибору кореневої директорії."""
    print("Відкриття діалогового вікна для вибору папки...")
    root = tk.Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory(title="Оберіть кореневу папку з каталогами solo, solo_1...")
    return directory_path

def analyze_data(root_dir):
    """
    Аналізує дані, створюючи єдиний список об'єктів (bboxes) для кожного каталогу.
    """
    def get_sort_key(dir_path):
        name = os.path.basename(dir_path)
        parts = name.split('_')
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1])
        return -1

    all_solo_dirs = glob.glob(os.path.join(root_dir, 'solo*'))
    
    if not all_solo_dirs:
        print(f"❌ У папці '{root_dir}' не знайдено каталогів з назвою 'solo...'.")
        return None
        
    solo_dirs = sorted(all_solo_dirs, key=get_sort_key)
    print(f"🔎 Знайдено {len(solo_dirs)} каталогів для аналізу: {[os.path.basename(d) for d in solo_dirs]}")

    all_stats = {}

    for solo_dir in solo_dirs:
        solo_name = os.path.basename(solo_dir)
        print(f"🔄 Обробка каталогу: {solo_name}...")
        annotation_files = glob.glob(os.path.join(solo_dir, '**', '*.json'), recursive=True)
        
        # --- ЗМІНА СТРУКТУРИ ДАНИХ ---
        bboxes = []

        for ann_file in annotation_files:
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for capture in data.get('captures', []):
                    image_filename = capture.get('filename')
                    if not image_filename:
                        continue
                    
                    image_path = os.path.join(os.path.dirname(ann_file), image_filename)

                    for annotation in capture.get('annotations', []):
                        if annotation.get('id') == 'bounding box':
                            for bbox_value in annotation.get('values', []):
                                dimension = bbox_value.get('dimension')
                                if dimension and len(dimension) == 2:
                                    # Зберігаємо повну інформацію про об'єкт
                                    bboxes.append({'path': image_path, 'width': dimension[0], 'height': dimension[1]})
            except json.JSONDecodeError:
                print(f"\n⚠️ Помилка декодування JSON у файлі {ann_file}")
            except Exception as e:
                print(f"\n⚠️ Неочікувана помилка при обробці файлу {ann_file}: {e}")
        
        all_stats[solo_name] = {'bboxes': bboxes}
        
    return all_stats

def create_interactive_plot(stats_to_plot, all_stats_with_paths, title, xlabel, data_type):
    """
    Створює інтерактивний графік з можливістю виділення діапазону для перегляду зображень.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(right=0.75)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(f"{xlabel} (логарифмічна шкала)")
    ax.set_ylabel('Кількість')
    ax.grid(True, which="both", linestyle='--', alpha=0.6)

    lines = {}
    labels = []
    
    hist_data = {name: [item[0] for item in data] for name, data in stats_to_plot.items()}
    all_values = [item for data in hist_data.values() for item in data if item > 0]
    
    if not all_values:
        ax.text(0.5, 0.5, 'Немає даних для відображення', ha='center', va='center', transform=ax.transAxes)
        plt.show()
        return

    min_val = np.floor(np.log10(min(all_values)))
    max_val = np.ceil(np.log10(max(all_values)))
    bins = np.logspace(min_val, max_val, num=75)

    for name, data in hist_data.items():
        filtered_data = [d for d in data if d > 0]
        if filtered_data:
            hist_type = 'bar' if len(hist_data) == 1 else 'step'
            alpha = 0.7 if len(hist_data) == 1 else 1.0
            _, _, patches = ax.hist(filtered_data, bins=bins, histtype=hist_type, label=name, linewidth=2, alpha=alpha)
            lines[name] = patches
            labels.append(name)
            
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.minorticks_off()
    
    fig.widgets = []

    if len(labels) > 1:
        checkbox_ax = plt.axes([0.78, 0.6, 0.18, 0.3])
        actives = [True] * len(labels)
        check = CheckButtons(checkbox_ax, labels, actives)
        
        def toggle_visibility(label):
            for patch in lines[label]:
                patch.set_visible(not patch.get_visible())
            fig.canvas.draw_idle()
        
        check.on_clicked(toggle_visibility)
        fig.widgets.append(check)
    
    info_ax = plt.axes([0.78, 0.2, 0.18, 0.3])
    info_ax.axis('off')

    avg_strings = ["Середні значення:", "-----------------"]
    for name, data_list in stats_to_plot.items():
        values = [item[0] for item in data_list]
        avg = np.mean(values) if values else 0
        avg_strings.append(f"{name}: {avg:.1f} px")
    
    info_ax.text(0.0, 1.0, "\n".join(avg_strings), verticalalignment='top', fontsize=10, 
                 fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    ax.legend(loc='upper left')

    def on_select(eclick, erelease):
        """
        Callback-функція, що збирає повну інформацію про об'єкти та передає у ImageViewer.
        """
        xmin, xmax = sorted((eclick.xdata, erelease.xdata))
        print(f"\n🖱️ Вибрано діапазон '{xlabel}': [{xmin:.2f}, {xmax:.2f}]")
        
        # --- ЗМІНА ЛОГІКИ: ЗБИРАЄМО ОБ'ЄКТИ, А НЕ ШЛЯХИ ---
        filtered_objects = []
        dimension_key = data_type.rstrip('s') # 'widths' -> 'width'
        
        for data in all_stats_with_paths.values():
            for bbox in data['bboxes']:
                if xmin <= bbox[dimension_key] <= xmax:
                    filtered_objects.append((bbox['path'], bbox['width'], bbox['height']))
        
        if filtered_objects:
            # Сортуємо для консистентного порядку
            filtered_objects.sort()
            print(f"🖼️ Знайдено {len(filtered_objects)} об'єктів. Відкриття переглядача...")
            # Передаємо список об'єктів, діапазон та тип розмірності
            ImageViewer(filtered_objects, (xmin, xmax), data_type)
        else:
            print("🤷 У вибраному діапазоні об'єктів не знайдено.")

    rs = RectangleSelector(ax, on_select, useblit=True, button=[1], minspanx=5, 
                           minspany=5, spancoords='pixels', interactive=True)
    fig.widgets.append(rs)

def plot_distributions(all_stats):
    """
    Готує дані для графіків на основі нової структури all_stats.
    """
    if not all_stats:
        print("Немає даних для побудови графіків.")
        return
    
    # Адаптуємо дані для функції create_interactive_plot
    width_stats = {name: [(bbox['width'], bbox['path']) for bbox in data['bboxes']] for name, data in all_stats.items()}
    height_stats = {name: [(bbox['height'], bbox['path']) for bbox in data['bboxes']] for name, data in all_stats.items()}

    if len(all_stats) > 1:
        create_interactive_plot(width_stats, all_stats, 'Розподіл ширин об\'єктів по каталогах', 'Ширина (пікселі)', 'widths')
        create_interactive_plot(height_stats, all_stats, 'Розподіл висот об\'єктів по каталогах', 'Висота (пікселі)', 'heights')

    # Адаптуємо дані для сумарних графіків
    total_widths = [(bbox['width'], bbox['path']) for data in all_stats.values() for bbox in data['bboxes']]
    total_heights = [(bbox['height'], bbox['path']) for data in all_stats.values() for bbox in data['bboxes']]
    
    summary_width_stats = {'Усі разом': total_widths}
    summary_height_stats = {'Усі разом': total_heights}

    create_interactive_plot(summary_width_stats, all_stats, 'Сумарний розподіл ширин для всіх об\'єктів', 'Ширина (пікселі)', 'widths')
    create_interactive_plot(summary_height_stats, all_stats, 'Сумарний розподіл висот для всіх об\'єктів', 'Висота (пікселі)', 'heights')
    
    print("\n✅ Аналіз завершено. Відображення інтерактивних графіків...")
    print("🖱️ Інструкція: затисніть ліву кнопку миші на графіку та протягніть, щоб виділити діапазон та переглянути зображення.")
    plt.show()

if __name__ == "__main__":
    root_directory = select_root_directory()
    
    if root_directory:
        print(f"Обрано директорію: {root_directory}")
        stats = analyze_data(root_directory)
        if stats:
            plot_distributions(stats)
    else:
        print("Директорію не обрано. Роботу завершено.")

print("Роботу завершено.")