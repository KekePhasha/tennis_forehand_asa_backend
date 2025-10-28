import json
from pathlib import Path

# import mmpretrain.models
from mmpose.apis import MMPoseInferencer
import numpy as np
import os

from utils.file_utils import FileSaver


class ViTPoseEstimator:
    """
    Class to extract 2D pose keypoints from a video using MMPose.
    The extracted keypoints are saved in a .npy file.
    The keypoints are in the format (x, y, score) for each of the 17 keypoints.
    """

    def __init__(self, saver: FileSaver, model='vitpose-s'):
        """
        Initialise the ViTPoseEstimator with the specified model and output directory.
        :param saver: An instance of FileSaver to handle saving keypoints.
        :param model: vitpose-s, vitpose-m, vitpose-l
        """
        # Initialise the MMPoseInferencer with the specified model and output directory from API.
        self.inferencer = MMPoseInferencer(pose2d=model, device='cpu')
        self.fileSaver = saver

    def extract_keypoints(self, video_path: str, save_name: str = None, save_visual: bool = False,
                          save_predictions: bool = False):
        """
        Extract keypoints from a video and save them to a .npy file.
        :param save_predictions:
        :param save_visual:
        :param video_path:
        :param save_name:
        :return: keypoints_array: A numpy array of shape (N, 17, 3) where N is the number of frames.
        N is the number of frames in the video. (52)
        17 is the number of keypoints. eye, ear, shoulder, elbow, wrist, hip, knee, ankle, etc.
        3 corresponds to (x, y, score) for each keypoint. x-coordinate, y-coordinate, and confidence score.
        e.g. keypoints_array[0] contains keypoints for the first frame - [[x1, y1, score1], [x2, y2, score2], ..., [x17, y17, score17]]
        """
        if save_name is None:
            save_name = os.path.splitext(os.path.basename(video_path))[0]

        run_dir = Path(self.fileSaver.prepare_run_dir(save_name))  # e.g., static/results/<save_name>/
        vis_dir = run_dir / "visualization"
        pred_dir = run_dir / "predictions"
        vis_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        result_generator = self.inferencer(
            video_path,
            black_background=True,
            vis_out_dir=str(vis_dir) if save_visual else None,
            pred_out_dir=str(pred_dir) if save_predictions else None,
            save_predictions=save_predictions,
            return_vis=False
        )
        keypoints_per_frame = []
        per_frame_meta = []

        # Loop through each frame in the video
        for result in result_generator:
            # Check if there is at least one prediction
            if result.get('predictions') and len(result['predictions'][0]) > 0:
                first_instance = result['predictions'][0][
                    0]  # First person in batch[0] returns for first person in video
                per_frame_meta.append(result['predictions'][0])  # Save full instances list for later
                kpts_xy = np.array(first_instance['keypoints'])  # shape: (17, 2)
                scores = np.array(first_instance['keypoint_scores'])  # shape: (17,)
                frame_kpts = np.hstack([kpts_xy, scores[:, None]])  # shape: (17, 3)
            else:
                # No detection — use empty keypoints
                frame_kpts = np.zeros((17, 3))  # Adjust shape if your model outputs more

            keypoints_per_frame.append(frame_kpts)

        keypoints_array = np.array(keypoints_per_frame)

        pred_json_path = None
        if save_predictions:
            # Keep only the first person points per frame to keep file small for frontend,
            # or store full instances list if you want multi-person later.
            merged = []
            for frame_instances in per_frame_meta:
                if frame_instances:
                    inst = frame_instances[0]
                    merged.append({
                        "keypoints": inst["keypoints"],  # (17,2)
                        "keypoint_scores": inst["keypoint_scores"],  # (17,)
                    })
                else:
                    merged.append({"keypoints": None, "keypoint_scores": None})

            pred_json_path = run_dir / f"{save_name}_predictions.json"
            with open(pred_json_path, "w") as f:
                json.dump(merged, f)

        skeleton_mp4 = None
        if save_visual:
            # Usually the tool saves a file named like the input video; grab the only mp4 there
            mp4s = list(vis_dir.glob("*.mp4"))
            if mp4s:
                skeleton_mp4 = mp4s[0]

        saved_paths = {"npy": self.fileSaver.save_keypoints(video_path, save_name, keypoints_array)}
        if skeleton_mp4:  saved_paths["skeleton_mp4"] = str(skeleton_mp4)
        if pred_json_path: saved_paths["pred_json"] = str(pred_json_path)
        return keypoints_array, saved_paths
