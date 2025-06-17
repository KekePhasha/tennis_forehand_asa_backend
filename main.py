import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from embedding.pose_embedding import PoseEmbedding
from training.siamese_dataset import SiamesePoseDataset
from models.siamese_model import SiameseModel
from training.loss import ContrastiveLoss
from pose_estimation.vitpose_extractor import ViTPoseEstimator
from utils.file_utils import FileSaver

if __name__ == '__main__':
    fileSaver = FileSaver()
    estimator = ViTPoseEstimator(fileSaver)
    poseEmbedding = PoseEmbedding(0.6)
    ### Training a Siamese Network for Pose Similarity ###
    video_root = 'dataset/VIDEO_RGB/forehand_openstands'
    keypoint_dir = 'dataset/keypoints'
    embedding_dir = 'dataset/embeddings'

    os.makedirs(keypoint_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)

    # step 1: Extract keypoints from all videos
    for label in ["positive", "negative"]:
        label_path = os.path.join(video_root, label)
        for video_file in os.listdir(label_path):
            if video_file.endswith('.avi'):
                video_path = os.path.join(label_path, video_file)
                keypoint_name = video_path.replace('.avi', '')
                save_name = os.path.splitext(video_file)[0]
                estimator.extract_keypoints(video_path, save_name)



    # step 2: Generate pose embedding
    ## -------- Generate pose embedding -------- ##
    # user1_embedding = poseEmbedding.generate_from_file("dataset/keypoints/user1_video_keypoints.npy")
    # user2_embedding = poseEmbedding.generate_from_file("dataset/keypoints/user2_video_keypoints.npy")
    # print("Pose embedding shape:", user1_embedding.shape)
    # print("Pose embedding shape:", user2_embedding.shape)
    #
    # ## Save the embeddings
    # user1_embedding_path = os.path.join('dataset/embeddings', 'user1_embedding.npy')
    # user2_embedding_path = os.path.join('dataset/embeddings', 'user2_embedding.npy')
    # os.makedirs('dataset/embeddings', exist_ok=True)
    # np.save(user1_embedding_path, user1_embedding)
    # np.save(user2_embedding_path, user2_embedding)
    #
    # pairs = [
    #     (user1_embedding_path, user2_embedding_path),
    #     (user1_embedding_path, user1_embedding_path),
    #     (user2_embedding_path, user2_embedding_path),
    # ]
    # labels = [0, 1, 1]  # 1 for similar, 0 for dissimilar

    ## -------- Load the dataset -------- ##
    # dataset = SiamesePoseDataset(pairs, labels)
    # print("Dataset length:", len(dataset))
    # loader = DataLoader(dataset, batch_size=3, shuffle=True, drop_last=True)
    #
    # model = SiameseModel()
    # loss_fn = ContrastiveLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    # epochs = 10
    # for epoch in range(epochs):
    #     for i, (emb1, emb2, label) in enumerate(loader):
    #         optimizer.zero_grad()
    #         distance = model.forward(emb1, emb2)
    #         loss = loss_fn(distance, label)
    #         loss.backward()
    #         optimizer.step()
    #
    #         if i % 10 == 0:
    #             print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")
    #
    # torch.save(model.state_dict(), 'models/siamese_model.pth')
    #
    # user_1_tensor = torch.from_numpy(user1_embedding).unsqueeze(0).float()
    # user_2_tensor = torch.from_numpy(user2_embedding).unsqueeze(0).float()
    #
    # model.eval()
    # with torch.no_grad():
    #     distance = model.forward(user_1_tensor, user_2_tensor).item()
    #
    # similarity_score = 1 / (1 + distance)
    #
    # print(f"Similarity score: {similarity_score:.2f}")



    ## -------- Display keypoints on a frame -------- ##
    # Open video and go to frame 70
    # cap = cv2.VideoCapture("dataset/VIDEO_RGB/forehand_openstands/p9_foreopen_s3.avi")
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 70)  # Seek to frame 70
    # success, frame = cap.read()
    # cap.release()
    #
    # if not success:
    #     raise ValueError("Could not read frame 70 from the video.")
    #
    # # Draw keypoints for frame 70
    # kp = keypoints[70]
    # for x, y, conf in kp:
    #     if conf > 0.5:
    #         cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    #
    # # Display
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.title("Pose keypoints (frame 70)")
    # plt.axis("off")
    # plt.show()

