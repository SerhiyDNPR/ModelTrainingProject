import numpy as np
import matplotlib.pyplot as plt

# Параметри
object_size_m = 1.0  # Розмір об'єкта в метрах
horizontal_resolution_px = 1920  # Горизонтальна роздільна здатність камери в пікселях
fov_h_deg = 60.0  # Горизонтальний кут огляду в градусах

# Конвертуємо кут огляду в радіани
fov_h_rad = np.deg2rad(fov_h_deg)

# Створюємо діапазон відстаней від 1 до 100 метрів
distances_m = np.linspace(1, 50, 400)

# Розраховуємо ширину сцени, яку бачить камера на кожній відстані
scene_width_m = 2 * distances_m * np.tan(fov_h_rad / 2)

# Розраховуємо розмір об'єкта в пікселях
object_size_px = (object_size_m / scene_width_m) * horizontal_resolution_px

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(distances_m, object_size_px)
plt.title("Залежність розміру об'єкта в пікселях від відстані")
plt.xlabel("Відстань до об'єкта (метри)")
plt.ylabel("Розмір об'єкта (пікселі)")
plt.grid(True)
plt.show() # Показує графік