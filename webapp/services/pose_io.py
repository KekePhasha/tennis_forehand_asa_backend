from utils.file_utils import FileSaver
from pose_estimation.vitpose_extractor import ViTPoseEstimator

class ViTPoseWrapper:
    def __init__(self):
        self.est = ViTPoseEstimator(FileSaver())

    def extract_keypoints(self, video_path: str, save_name: str, as_tensor=False, device="cpu"):
        # Keep your original method signature/behavior
        return self.est.extract_keypoints(video_path, save_name=save_name, as_tensor=as_tensor, device=device)
