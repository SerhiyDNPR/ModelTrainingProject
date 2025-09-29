from multiprocessing import freeze_support
import config
from managers import TrainingManager

if __name__ == '__main__':
    # freeze_support() потрібен для коректної роботи multiprocessing у Windows
    freeze_support()
    
    # Створюємо екземпляр менеджера та передаємо йому конфігурацію
    manager = TrainingManager(config)
    
    # Запускаємо процес
    manager.run()