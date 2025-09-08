import os
from embedding.pose_embedding import PoseEmbedding
from pose_estimation.vitpose_extractor import ViTPoseEstimator
from utils.file_utils import FileSaver

class KeypointExtract:
    def __init__(self,
                 video_root='dataset/VIDEO_RGB/forehand_openstands',
                 keypoint_dir='dataset/keypoints',
                 embedding_dir='dataset/embeddings',
                 confidence_threshold=0.6):
        self.video_root = video_root
        self.keypoint_dir = keypoint_dir
        self.embedding_dir = embedding_dir
        self.file_saver = FileSaver(base_dir='dataset')
        self.estimator = ViTPoseEstimator(self.file_saver)
        self.embedder = PoseEmbedding(confidence_threshold=confidence_threshold)

    def extract_all_keypoints(self):
        """
        Extract keypoints from all videos in the dataset and save them to .npy files.
        """
        for label in ['positive', 'negative']:
            label_path = os.path.join(self.video_root, label)
            if not os.path.exists(label_path):
                continue
            for video_file in os.listdir(label_path):
                if video_file.endswith('.avi'):
                    video_path = os.path.join(label_path, video_file)
                    save_name = os.path.splitext(video_file)[0]
                    self.estimator.extract_keypoints(video_path, save_name)

    def generate_all_embeddings(self):
        """
        Generate embeddings for all keypoints in the dataset.
        """
        for label in ['positive', 'negative', 'test']:
            label_dir = os.path.join(self.keypoint_dir, label)
            if not os.path.exists(label_dir):
                continue
            for file in os.listdir(label_dir):
                if file.endswith('.npy'):
                    kp_path = os.path.join(label_dir, file)
                    embedding = self.embedder.generate_from_file(kp_path)
                    self.file_saver.save_embedding(kp_path, embedding)

    def generate_pairs(self):
        """
        Generate pairs of embeddings for layers the Siamese network.
        :return: pairs: List of tuples containing paths to pairs of embeddings.
        """
        # Group embeddings
        pos_dir = os.path.join(self.embedding_dir, 'positive')
        neg_dir = os.path.join(self.embedding_dir, 'negative')

        # List all .npy files in the positive and negative directories
        pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.endswith('.npy')]
        neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.endswith('.npy')]

        pairs = []
        labels = []

        # Positive pairs (same class)
        for i in range(len(pos_files)):
            for j in range(i + 1, len(pos_files)):
                pairs.append((pos_files[i], pos_files[j]))
                labels.append(1)

        for i in range(len(neg_files)):
            for j in range(i + 1, len(neg_files)):
                pairs.append((neg_files[i], neg_files[j]))
                labels.append(1)

        # Negative pairs (different classes)
        for pf in pos_files:
            for nf in neg_files:
                pairs.append((pf, nf))
                labels.append(0)

        return pairs, labels

