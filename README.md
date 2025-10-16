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


python train.py \
    --backend pure \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-4 \
    --margin 1.5 \
    --embed_dim 64 \
    --use_bn \
    --use_dropout
  

python train.py --backend pure --epochs 50 --batch_size 4 --lr 5e-4 --margin 0.5 --embed_dim 64

python train.py   --backend r3d_18   --epochs 10   --batch_size 4   --lr 1e-3