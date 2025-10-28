from utils.file_utils import FileSaver
from pose_estimation.vitpose_extractor import ViTPoseEstimator

class ViTPoseWrapper:
    def __init__(self):
        self.est = ViTPoseEstimator(FileSaver())
        self.saver = FileSaver(base_dir='dataset', public_mount='/static')

    def extract_keypoints(self, video_path: str, save_name: str, save_visual: bool = False, save_predictions: bool = False):
        # Keep your original method signature/behaviour
        return self.est.extract_keypoints(video_path, save_name=save_name, save_visual=save_visual, save_predictions=save_predictions)

    def paths_to_urls(self, saved_paths: dict) -> dict:
        # convert any path value to a URL using FileSaver.as_url
        out = {}
        for k, v in (saved_paths or {}).items():
            out[k] = self.saver.as_url(v)
        return out
