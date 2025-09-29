class Prediction:
    """
    Уніфікований клас для зберігання  результату детекції.
    """
    def __init__(self, box, score, class_id, class_name, track_id=None):
        self.box = box  # Координати [x1, y1, x2, y2]
        self.score = score  # Впевненість (0.0 to 1.0)
        self.class_id = class_id  # 0-індексований ID класу
        self.class_name = class_name  # Назва класу
        self.track_id = track_id  # Опціональний ID для трекінгу