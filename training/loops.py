import math
import torch
import torch.nn.functional as F

def l2_normalize_rows_py(list_of_lists):
    normalized = []
    for row in list_of_lists:
        norm = math.sqrt(sum(v*v for v in row))
        normalized.append([v/norm for v in row] if norm > 0 else row[:])
    return normalized

class ContrastiveLossTorch(torch.nn.Module):
    """labels: 0=similar, 1=dissimilar"""
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    def forward(self, distance_tensor: torch.Tensor, label_tensor: torch.Tensor):
        labels = label_tensor.float()
        positive_term  = 0.5 * (1 - labels) * distance_tensor.pow(2)
        negative_term  = 0.5 * labels * F.relu(self.margin - distance_tensor).pow(2)
        return (positive_term + negative_term).mean()

def train_epoch_pure(model, data_loader, learning_rate=5e-4, margin=1.0, momentum=0.9):
    running_loss, num_steps = 0.0, 0
    for left_vec, right_vec, label_tensor in data_loader:
        left_list  = left_vec.numpy().astype(float).tolist()
        right_list = right_vec.numpy().astype(float).tolist()
        label_list = label_tensor.numpy().astype(int).tolist()
        loss = model.train_batch(left_list, right_list, label_list,
                                 lr=learning_rate, margin=margin)
        running_loss += loss; num_steps += 1
    return running_loss / max(1, num_steps)

@torch.no_grad()
def eval_epoch_pure(model, data_loader, margin=1.0):
    correct, total = 0, 0
    for left_vec, right_vec, label_tensor in data_loader:
        left_list  = left_vec.numpy().astype(float).tolist()
        right_list = right_vec.numpy().astype(float).tolist()
        label_list = label_tensor.numpy().astype(int).tolist()
        distance_list = model.distances(left_list, right_list)
        for dist, lbl in zip(distance_list, label_list):
            pred_is_similar = dist < margin
            correct += int(pred_is_similar == (lbl == 0)); total += 1
    return correct / max(1, total)

def train_epoch_torch(model, data_loader, optimizer, margin=1.0, device="cpu"):
    model.train()
    loss_fn = ContrastiveLossTorch(margin)
    running_loss, num_steps = 0.0, 0
    for left_tensor, right_tensor, label_tensor in data_loader:
        left_tensor  = left_tensor.to(device)
        right_tensor = right_tensor.to(device)
        label_tensor = label_tensor.to(device)

        optimizer.zero_grad()
        out = model(left_tensor, right_tensor)

        # Standardize what we take as "distance"
        if isinstance(out, dict):
            # preferred if your model returns a dict
            distance_tensor = out["distance"]  # change key if different
        elif isinstance(out, tuple):
            # common patterns: (z1, z2, distance), or extra items at the end
            if len(out) >= 3:
                # assume distance is last or the 3rd element; pick last scalar-like tensor
                # safest: take the last element
                distance_tensor = out[-1]
            else:
                raise ValueError(f"Unexpected tuple length from model: {len(out)}")
        else:
            # if model returns distance directly as a tensor
            distance_tensor = out

        loss = loss_fn(distance_tensor, label_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running_loss += loss.item(); num_steps += 1
    return running_loss / max(1, num_steps)

@torch.no_grad()
def eval_epoch_torch(model, data_loader, margin=1.0, device="cpu"):
    model.eval()
    correct, total = 0, 0

    for left_tensor, right_tensor, label_tensor in data_loader:
        left_tensor  = left_tensor.to(device)
        right_tensor = right_tensor.to(device)
        label_tensor = label_tensor.to(device)

        out = model(left_tensor, right_tensor)

        # Standardize to a 1-D distance tensor: shape (B,)
        if isinstance(out, dict):
            dist = out["distance"]
        elif isinstance(out, (tuple, list)):
            dist = out[-1]               # last item should be distance
        else:
            dist = out                   # model returns distance directly

        if dist.ndim > 1:                # e.g., model returned (B, D) embeddings/distances
            dist = dist.norm(dim=1)      # collapse to per-sample scalar

        dist = dist.view(-1)             # ensure (B,)
        preds_similar = dist < margin    # True if predicted similar

        # labels: 0 = similar, 1 = dissimilar
        target_similar = (label_tensor == 0).view(-1)

        correct += (preds_similar == target_similar).sum().item()
        total   += target_similar.numel()

    return correct / max(1, total)

