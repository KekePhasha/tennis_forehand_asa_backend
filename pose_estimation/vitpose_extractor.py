from mmpose.apis import MMPoseInferencer
import numpy as np
import os


# MMPose Inferencer. It’s a unified inferencer interface for pose estimation task, currently including: Pose2D.
# and it can be used to perform 2D keypoint detection.
class ViTPoseEstimator:
    def __init__(self, output='dataset/keypoints', model='vitpose-small'):
        # Initialize the MMPoseInferencer with the specified model and output directory from API.
        self.inferencer = MMPoseInferencer(pose2d=model, det_model='YOLOv8')
        self.output_dir = output
        os.makedirs(output, exist_ok=True)

    def extract_keypoints(self, video_path: str, save_name: str = None):
        if save_name is None:
            save_name = os.path.splitext(os.path.basename(video_path))[0]

        keypoints_per_frame = []
        result_generator = self.inferencer(video_path, show=False)

        for result in result_generator:
            frame_kpts = None
            if result.get("predictions"):
                kpts = result["predictions"][0]["keypoints"]
                frame_kpts = np.array(kpts)
            else:
                # If no keypoints are detected, append None
                frame_kpts = np.full((17, 3), np.nan)

            keypoints_per_frame.append(frame_kpts)

        keypoints_array = np.array(keypoints_per_frame)
        save_path = os.path.join(self.output_dir, f'{save_name}.npy')
        np.save(save_path, keypoints_array)

        print(f'Keypoints saved to {save_path}')
        return keypoints_array
