from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Union


class YoloDetection:
    def __init__(
        self,
        model_name: str = "models/yolov8m.pt",
        device: str = "cpu",
        conf_threshold: float = 0.3,
    ):
        self.model = YOLO(model_name).to(device)
        self.device = device
        self.conf_threshold = conf_threshold

    def infer(
        self,
        image: np.ndarray,
        return_boxes: bool = False,
    ) -> Union[
        List[Tuple[str, float, Tuple[int, int]]],
        List[Tuple[str, float, Tuple[int, int], np.ndarray]],
    ]:
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy array")

        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Input must be an RGB image (H, W, 3)")

        if image.shape[2] > 3:
            image = image[:, :, :3]
        image = image[:, :, ::-1]

        results = self.model.predict(image, verbose=False, conf=self.conf_threshold)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()

            x_center = int((xyxy[0] + xyxy[2]) / 2)
            y_center = int((xyxy[1] + xyxy[3]) / 2)

            class_name = self.model.names[cls_id]

            if return_boxes:
                detections.append((class_name, conf, (x_center, y_center), xyxy))
            else:
                detections.append((class_name, conf, (x_center, y_center)))

        return detections

    def __call__(self, image: np.ndarray, return_boxes: bool = False):
        return self.infer(image, return_boxes)
