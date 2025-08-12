import torch
import numpy as np
import time


class MiDasEstimation:
    def __init__(self, model_name: str = "MiDaS_small", device: str = "cpu"):
        self.model = torch.hub.load("intel-isl/MiDaS", model_name).to(device)
        self.device = device
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_name in ["DPT_Large", "DPT_Hybrid", "DPT_Lite"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def infer(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Input must be RGB Image (H, W, 3)")

        original_height, original_width = image.shape[:2]

        img = self.transform(image).to(self.device)
        with torch.no_grad():
            pred = self.model(img)

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(original_height, original_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = pred.cpu().numpy()
        depth_map_normalized = (depth_map - depth_map.min()) / (
            depth_map.max() - depth_map.min() + 1e-8
        )
        return depth_map_normalized

    def __call__(self, image: np.ndarray):
        return self.infer(image)


if __name__ == "__main__":
    HEIGHT = 480
    WIDTH = 640
    image = np.random.rand(HEIGHT, WIDTH, 3)
    model = MiDasEstimation(model_name="MiDaS_small", device="cpu")
    start = time.time()
    output = model(image)
    end = time.time() - start
    print("runtime", end)
