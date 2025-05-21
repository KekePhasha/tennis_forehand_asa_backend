import cv2
import numpy as np
from matplotlib import pyplot as plt

from embedding.pose_embedding import PoseEmbedding
from pose_estimation.vitpose_extractor import ViTPoseEstimator

if __name__ == '__main__':
    #step 1: Extract keypoints from video
    # estimator = ViTPoseEstimator()
    # user1_keypoints = estimator.extract_keypoints( 'dataset/VIDEO_RGB/forehand_openstands/p9_foreopen_s3.avi', save_name='user1_video_keypoints')
    # user2_keypoints = estimator.extract_keypoints( 'dataset/VIDEO_RGB/forehand_openstands/p9_foreopen_s3.avi', save_name='user2_video_keypoints')

    # Load keypoints from .npy
    # keypoints = np.load("dataset/keypoints/sample_video_keypoints.npy")
    # print("Shape:", keypoints.shape)
    # print("Example keypoint:", keypoints[70][0])  # Show one from frame 70

    #step 2: Generate pose embedding
    ## -------- Generate pose embedding -------- ##
    poseEmbedding = PoseEmbedding(confidence_threshold=0.6)
    user1_embedding = poseEmbedding.generate_from_file("dataset/keypoints/user1_video_keypoints.npy")
    user2_embedding = poseEmbedding.generate_from_file("dataset/keypoints/user2_video_keypoints.npy")
    print("Pose embedding shape:", user1_embedding.shape)
    print("Pose embedding shape:", user2_embedding.shape)



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

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
