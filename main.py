# Press the green button in the gutter to run the script.
from pose_estimation.vitpose_extractor import ViTPoseEstimator

if __name__ == '__main__':
    estimator = ViTPoseEstimator
    keypoints = estimator.extract_keypoints( 'dataset/videos/sample_video.mp4', save_name='sample_video_keypoints')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
