import numpy as np


class PoseEmbedding:
    """
    Class to generate a fixed-size embedding for a sequence of pose keypoints.
    """
    def __init__(self, confidence_threshold=0.8):
        """
        Initialize the PoseEmbedding class with a confidence threshold.
        :param confidence_threshold:
        """
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def normalize_pose(keypoints):
        """
        Normalize the pose keypoints to a fixed size and center them.
        :param keypoints: A single frame of keypoints with shape (17, 3).
        :return: Normalized keypoints with shape (17, 3).
        """
        # getting the left shoulder keypoint and return the x and y coordinates Shape: (71, 17, 3)
        left_shoulder = keypoints[5, :2]
        right_shoulder = keypoints[6, :2]
        center = (left_shoulder + right_shoulder) / 2
        # Calculate the distance between the shoulders
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6

        # Normalize the keypoints
        normalized_keypoints = (keypoints[:, :2] - center) / shoulder_distance
        confidence = keypoints[:, 2:3]
        return np.concatenate((normalized_keypoints, confidence), axis=1)

    def generate_embedding(self, keypoints_seq):
        """
        Generate a fixed-size embedding for a sequence of keypoints.
        :param keypoints_seq: A sequence of keypoints with shape (N, 17, 3).
        :return: A fixed-size embedding with shape (N, 34).
        What this function does:
        It processes a sequence of keypoints, normalizes each frame, and returns a fixed-size embedding.
        """
        valid_frames = []

        for frame in keypoints_seq:
            # Check if any keypoint has a confidence score above the threshold
            if np.mean(frame[:, 2] > self.confidence_threshold):
                normalized_pose = self.normalize_pose(frame)
                valid_frames.append(normalized_pose.flatten())

        if not valid_frames:
            return np.zeros((17, 3)) # Return a zero vector if no valid frames

        return np.mean(valid_frames, axis=0)

    def generate_from_file(self, keypoints_file: str):
        """
        Generate a fixed-size embedding for a sequence of keypoints from a file.
        :param keypoints_file: Path to the .npy file containing keypoints.
        :return: A fixed-size embedding with shape (N, 34).
        """
        keypoints_seq = np.load(keypoints_file)
        return self.generate_embedding(keypoints_seq)

