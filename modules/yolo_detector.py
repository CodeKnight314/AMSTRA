from ultralytics import YOLO
import numpy as np
from typing import List, Union
import cpuinfo
from PIL import Image
import coremltools as ct
import cv2
import os


class YoloDetection:
    def __init__(
        self, model_name="models/yolov8n.pt", device="cpu", conf_threshold=0.25
    ):
        self.model = YOLO(model_name, task="detect")
        self.device = device
        self.conf_threshold = conf_threshold

    def infer(self, image: np.ndarray, return_boxes: bool):
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Input must be a BGR image (H, W, 3)")
        if image.shape[2] > 3:
            image = image[:, :, :3]

        H, W = image.shape[:2]
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            imgsz=(H, W),
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            class_name = self.model.names[cls_id]
            if return_boxes:
                detections.append(
                    (class_name, conf, (x_center, y_center), np.array([x1, y1, x2, y2]))
                )
            else:
                detections.append((class_name, conf, (x_center, y_center)))

        return detections

    def __call__(self, image: np.ndarray, return_boxes: bool = True):
        return self.infer(image, return_boxes)


class YoloDetectionCoreML:
    def __init__(
        self, model_name="models/yolov8m.pt", device="cpu", conf_threshold=0.25
    ):
        mlpackage_path = model_name.replace(".pt", ".mlpackage")
        if not os.path.exists(mlpackage_path):
            yolo_model = YOLO(model_name)
            export_path = yolo_model.export(
                format="coreml", imgsz=640, nms=True, half=False, int8=False
            )
            mlpackage_path = export_path
            self.class_names = yolo_model.names
        else:
            self.class_names = YOLO(model_name).names

        self.model = ct.models.MLModel(mlpackage_path)
        self.device = device
        self.conf_threshold = conf_threshold

    def infer(self, image: Union[np.ndarray, Image.Image], return_boxes: bool):
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError("Input must be a BGR image (H, W, 3)")
            H, W = image.shape[:2]
            pil_img = Image.fromarray(
                cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
            ).resize((640, 640))
        else:
            pil_img = image.resize((640, 640))
            W, H = pil_img.size

        preds = self.model.predict(
            {"image": pil_img, "confidenceThreshold": self.conf_threshold}
        )
        return self._parse_outputs(preds, (H, W), return_boxes)

    def _parse_outputs(
        self, outputs: dict, original_size: List[int], return_boxes: bool
    ):
        H, W = original_size
        confs = outputs["confidence"]
        coords = outputs["coordinates"]
        boxes = []
        for idx in range(confs.shape[0]):
            class_idx = int(np.argmax(confs[idx]))
            conf = float(confs[idx][class_idx])

            cx, cy, w, h = coords[idx]
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H

            center_x = cx * W
            center_y = cy * H

            name = self.class_names.get(class_idx, str(class_idx))
            if return_boxes:
                boxes.append((name, conf, (center_x, center_y), [x1, y1, x2, y2]))
            else:
                boxes.append((name, conf, (center_x, center_y)))
        return boxes

    def __call__(self, image: np.ndarray, return_boxes: bool = True):
        return self.infer(image, return_boxes)


class YoloDetectionOpenVINO:
    def __init__(
        self, model_name="models/yolov8m.pt", device="cpu", conf_threshold=0.35
    ):
        ov_dir = model_name.replace(".pt", "_openvino_model")
        if not os.path.exists(ov_dir):
            yolo_model = YOLO(model_name)
            export_path = yolo_model.export(
                format="openvino", task="detect", imgsz=[480, 640], nms=True
            )
            ov_dir = export_path
        self.model = YOLO(ov_dir)
        self.device = device
        self.conf_threshold = conf_threshold

    def infer(self, image: np.ndarray, return_boxes: bool = True):
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Input must be a BGR image (H, W, 3)")
        H, W = image.shape[:2]

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            imgsz=(H, W),
            verbose=False,
        )

        boxes = []
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                name = names[cls_id]

                if return_boxes:
                    boxes.append((name, conf, (center_x, center_y), [x1, y1, x2, y2]))
                else:
                    boxes.append((name, conf, (center_x, center_y)))
        return boxes

    def __call__(self, image: np.ndarray, return_boxes: bool = True):
        return self.infer(image, return_boxes)


class YoloDetectionMain:
    def __init__(
        self,
        model_name: str = "models/yolov8m.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
    ):
        cpu_type = self._detect_cpu()
        if cpu_type == "Apple Silicon":
            self.model = YoloDetectionCoreML(model_name, device, conf_threshold)
        elif cpu_type == "Unknown":
            self.model = YoloDetection(model_name, device, conf_threshold)
        else:
            self.model = YoloDetectionOpenVINO(model_name, device, conf_threshold)

    def _detect_cpu(self):
        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw", "").lower()

        if "apple" in brand:
            return "Apple Silicon"
        elif "intel" in brand:
            return "Intel"
        elif "amd" in brand:
            return "AMD"
        elif "arm" in brand:
            return "ARM"

        return "Unknown"

    def infer(self, image: np.ndarray, return_boxes: bool = True):
        return self.model(image, return_boxes)

    def __call__(self, image: np.ndarray, return_boxes: bool = True):
        return self.infer(image, return_boxes)
