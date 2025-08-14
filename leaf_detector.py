from ultralytics import YOLO

MODEL_PATH = './yolo-runs/yolo11n-leaf-detector/detect/train/weights/best.pt'


class LeafDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH, verbose=False)

    def detect(self, image, conf_threshold=0.5):
        """
        Return bounding boxes for leaves in the image.

        Args:
            image (numpy.ndarray or str): The input image in BGR format as a NumPy array.
            conf_threshold (float, optional): Minimum confidence score required to keep a detection.
                Defaults to 0.5.

        Returns:
            list[tuple[tuple[int, int, int, int], float]]:
                A list of detections, where each detection is a tuple:
                    - coords (tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2).
                    - confidence (float): Detection confidence score.
        """
        result = self.model(image, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            confidence = box.conf.item()
            if confidence >= conf_threshold:
                coords = tuple(map(int, box.xyxy[0].tolist()))
                boxes.append((coords, confidence))
        return boxes
