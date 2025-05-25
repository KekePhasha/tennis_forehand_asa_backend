import torch
import torch.nn as nn

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
    def __init__(self, margin=2.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        """
        Contrastive loss function.
        :param distance: Euclidean distance between the two embeddings.
        :param label: Label indicating whether the pair is similar (1) or dissimilar (0).
        :return: Computed contrastive loss.
        """
        print("Distance:", distance)

        similar_loss = (1 - label) * distance ** 2
        dissimilar_loss = label * (torch.clamp(self.margin - distance, min=0.0) ** 2)
        loss = torch.mean(similar_loss + dissimilar_loss)
        return loss