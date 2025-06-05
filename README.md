# Tennis Forehand Action Similarity Assessment (ASA)

This project uses 2D human pose estimation and a Siamese neural network to assess the similarity between a user's tennis forehand stroke and that of an expert. The system extracts pose keypoints from video, generates embeddings, and provides feedback on technique alignment.

---

##  Features
- Upload user and expert stroke videos
- Extract pose keypoints using **ViTPose**
- Normalize and embed poses into fixed-size vectors
- Train a **Siamese model** with **contrastive loss**
- Compute similarity score between strokes
- Display feedback and annotated visual comparison

---

## Getting Started
### Prerequisites
- Python 3.8+
- PyTorch 1.7+
- OpenCV
- NumPy
- MMPose/mmcv
- Matplotlib
- Flask

```bash
pip install -r requirements.txt
```

### Resources
- VitPose -  https://mmpose.readthedocs.io/en/latest/user_guides/inference.html
-  torch.nn - https://docs.pytorch.org/docs/stable/nn.html