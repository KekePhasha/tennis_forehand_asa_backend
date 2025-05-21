import torch
import torch.nn as nn
import torch.nn.functional as f


class SiameseModel(nn.Module):
    def __init__(self, input_dim=51):
        super(SiameseModel, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x1, x2):
        f1 = self.embedding_net(x1)
        f2 = self.embedding_net(x2)
        return f.pairwise_distance(f1, f2)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    The loss function is defined as:
    L = (1 - y) * d^2 + y * max(0, m - d)^2
    where:
    - L is the loss
    - y is the label (1 if similar, 0 if dissimilar)
    - d is the distance between the two embeddings
    - m is the margin (default is 1.0)
    This loss function encourages the model to minimize the distance between similar pairs and maximize the distance between dissimilar pairs.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        """
        Contrastive loss function.
        :param distance:
        :param label:
        :return:
        """
        loss = torch.mean((1 - label) * torch.pow(distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss