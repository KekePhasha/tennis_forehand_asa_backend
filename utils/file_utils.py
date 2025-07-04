import os
import numpy as np

class FileSaver:
    """
    Class to save numpy arrays to files.
    """

    def __init__(self,base_dir='dataset'):
        """
        Initialize the FileSaver with a base directory.
        :param base_dir: Directory to save the files.
        """
        self.base_dir = base_dir
        self.keypoint_dir = os.path.join(base_dir, 'keypoints')
        self.embedding_dir = os.path.join(base_dir, 'embeddings')

        os.makedirs(self.keypoint_dir, exist_ok=True)
        os.makedirs(self.embedding_dir, exist_ok=True)

    def _get_label_folder(self, path):
        """
        Determine the label folder based on the path.
        :param path: Path to the video file.
        :return: 'positive', 'negative', 'test', or 'unsorted'.
        """
        parts = os.path.normpath(path).split(os.sep)
        for p in parts:
            if p.lower() in ['positive', 'negative', 'test']:
                return p.lower()
        return 'unsorted'

    def save_keypoints(self,video_path, save_name, keypoints):
        """
        Save keypoints to a .npy file.
        :param video_path: Path to the video file.
        :param save_name: Name to save the keypoints file.
        :param keypoints: Numpy array of keypoints with shape (N, 17, 3).
        """
        label = self._get_label_folder(video_path)
        save_dir = os.path.join(self.keypoint_dir, label)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f'{save_name}.npy')
        np.save(save_path, keypoints)
        return save_path

    def save_embedding(self, keypoints_path, embedding):
        """
        Save the embedding to a .npy file.
        :param keypoints_path:
        :param embedding:
        :return:
        """
        rel_path = os.path.relpath(keypoints_path, self.keypoint_dir)
        label = os.path.dirname(rel_path)
        save_dir = os.path.join(self.embedding_dir, label)
        os.makedirs(save_dir, exist_ok=True)

        save_name = os.path.splitext(os.path.basename(keypoints_path))[0] + '_embedding.npy'
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, embedding)
        return save_path
