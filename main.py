import torch
from torch.utils.data import DataLoader
from training.siamese_dataset import SiamesePoseDataset
from models.siamese_model import SiameseModel
from training.loss import ContrastiveLoss
from training.train_model import TrainModel

if __name__ == '__main__':
    trainer = TrainModel()
    # trainer.extract_all_keypoints()
    # trainer.generate_all_embeddings()
    pairs, labels = trainer.generate_pairs()
    dataset = SiamesePoseDataset(pairs, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SiameseModel()
    loss_fn = ContrastiveLoss()
    # Adam optimizer - used to optimize the model parameters is this backpropagation step?
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 40
    for epoch in range(epochs):
        for i, (emb1, emb2, label) in enumerate(loader):
            optimizer.zero_grad()
            distance = model.forward(emb1, emb2)
            loss = loss_fn(distance, label)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

    torch.save(model.state_dict(), 'models/siamese_model.pth')
    print("Training completed. Model saved to 'models/siamese_model.pth'")

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

