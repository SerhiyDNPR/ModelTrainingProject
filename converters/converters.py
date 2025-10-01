import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
import stat

def remove_readonly(func, path, exc_info):
    """
    Обробник помилок для shutil.rmtree, який видаляє прапорець "тільки для читання"
    і повторює спробу видалення у випадку помилки доступу.
    """
    # exc_info[1] містить екземпляр помилки
    if isinstance(exc_info[1], PermissionError):
        # Змінюємо права доступу на "запис" і пробуємо ще раз
        os.chmod(path, stat.S_IWRITE)
        func(path) # Повторюємо оригінальну функцію (напр., os.remove)
    else:
        # Якщо це інша помилка, просто викидаємо її
        raise

class BaseDataConverter(ABC):
    """Абстрактний базовий клас для всіх конвертерів даних."""
    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        if not self.source_dir.exists() and source_dir: # Перевіряємо, чи шлях не порожній
            raise FileNotFoundError(f"ПОМИЛКА: Директорія з вихідними даними не знайдена: {self.source_dir}")

    @abstractmethod
    def prepare_data(self):
        """Головний метод, що запускає процес конвертації."""
        pass

    def _natural_sort_key(self, s):
        """Ключ для "природного" сортування рядків."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s.name)]