
class SahiResultAdapter:
    # Словник з налаштуваннями, що передається в процесор
    SAHI_SETTINGS = {
        'CONF_THRESHOLD': 0.3,
        'SLICE_HEIGHT': 480,
        'SLICE_WIDTH': 640,
        'OVERLAP_HEIGHT_RATIO': 0.2,
        'OVERLAP_WIDTH_RATIO': 0.2,
    }


    """Адаптує результат SAHI до формату, схожого на результат YOLO."""
    def __init__(self, sahi_predictions):
        self.sahi_predictions = sahi_predictions
        self.boxes = self._convert_to_yolo_format()

    def _convert_to_yolo_format(self):
        yolo_boxes = []
        for pred in self.sahi_predictions:
            box_data = {
                'xyxy': [pred.bbox.to_xyxy()],
                'conf': [pred.score.value],
                'cls': [pred.category.id],
                'id': None
            }
            yolo_boxes.append(type('YoloBox', (), box_data)())
        return yolo_boxes