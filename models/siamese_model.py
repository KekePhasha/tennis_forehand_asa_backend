import torch.nn as nn
import torch.nn.functional as f

class SiameseModel(nn.Module):
    """
    Siamese network model for learning pose embeddings.
    This model consists of a shared embedding network that processes two input embeddings
    and outputs the distance between them.
    """
    def __init__(self, input_dim=51):
        super(SiameseModel, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x1, x2):
        """
        Forward pass for the Siamese network.
        :param x1: First input embedding.
        :param x2: Second input embedding.
        :return: Euclidean distance between the two embeddings.
        """
        f1 = self.embedding_net(x1)
        f2 = self.embedding_net(x2)
        return f.pairwise_distance(f1, f2)

        #todo
        # Should I use library like torch.nn.functional.pairwise_distance?, to calculate the distance between two embeddings