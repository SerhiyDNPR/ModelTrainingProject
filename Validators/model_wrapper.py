# model_wrapper.py

from abc import ABC, abstractmethod

class ModelWrapper(ABC):
    """
    Абстрактний базовий клас (інтерфейс) для всіх обгорток моделей.
    Визначає методи, які має реалізувати кожна конкретна обгортка.
    """
    def __init__(self, class_names, device):
        self.model = None
        self.device = device
        self.class_names = class_names
        print(f"Обгортку ініціалізовано для {len(self.class_names)} класів на пристрої {self.device}.")

    @abstractmethod
    def load(self, model_path):
        """
        Абстрактний метод для завантаження ваг моделі.
        Має бути реалізований у кожному дочірньому класі.
        """
        pass

    @abstractmethod
    def predict(self, frame, conf_threshold):
        """
        Абстрактний метод для виконання інференсу на одному кадрі.
        Має повертати список об'єктів Prediction.
        """
        pass