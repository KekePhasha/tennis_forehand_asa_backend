from mmpose.apis import MMPoseInferencer
import numpy as np
import os


# MMPose Inferencer. It’s a unified inferencer interface for pose estimation task, currently including: Pose2D.
# and it can be used to perform 2D keypoint detection.
class ViTPoseEstimator:
    """
    Class to extract 2D pose keypoints from a video using MMPose.
    The extracted keypoints are saved in a .npy file.
    The keypoints are in the format (x, y, score) for each of the 17 keypoints.
    """

    def __init__(self, output='dataset/keypoints', model='vitpose-s'):
        """
        Initialize the ViTPoseEstimator with the specified model and output directory.
        :param output: Directory to save the extracted keypoints.
        :param model: vitpose-s, vitpose-m, vitpose-l
        """
        # Initialize the MMPoseInferencer with the specified model and output directory from API.
        self.inferencer = MMPoseInferencer(pose2d=model, det_model='yolox_tiny_8x8_300e_coco', device='cpu')
        self.output_dir = output
        os.makedirs(output, exist_ok=True)

    def extract_keypoints(self, video_path: str, save_name: str = None):
        """
        Extract keypoints from a video and save them to a .npy file.
        :param video_path:
        :param save_name:
        :return: keypoints_array: A numpy array of shape (N, 17, 3) where N is the number of frames.
        """
        if save_name is None:
            save_name = os.path.splitext(os.path.basename(video_path))[0]

        keypoints_per_frame = []
        result_generator = self.inferencer(video_path, show=False)

        for result in result_generator:
            frame_kpts = None
            # Check if there is at least one prediction
            if result.get('predictions') and len(result['predictions'][0]) > 0:
                first_instance = result['predictions'][0][0]  # First person in batch[0]
                kpts_xy = np.array(first_instance['keypoints'])  # shape: (17, 2)
                scores = np.array(first_instance['keypoint_scores'])  # shape: (17,)
                frame_kpts = np.hstack([kpts_xy, scores[:, None]])  # shape: (17, 3)
            else:
                # No detection — use empty keypoints
                frame_kpts = np.zeros((17, 3))  # Adjust shape if your model outputs more

            keypoints_per_frame.append(frame_kpts)

        keypoints_array = np.array(keypoints_per_frame)
        save_path = os.path.join(self.output_dir, f'{save_name}.npy')
        np.save(save_path, keypoints_array)

        return keypoints_array
