from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Helper: build index tensors for body-part pooling
# ------------------------------------------------------------
def build_part_index_tensor(body_parts: dict[str, list[int]], J: int, device):
    parts = list(body_parts.values())
    max_len = max(len(idx) for idx in parts)
    P = len(parts)
    idx_tensor = torch.full((P, max_len), fill_value=-1, dtype=torch.long, device=device)
    mask_tensor = torch.zeros((P, max_len), dtype=torch.bool, device=device)
    for p, idxs in enumerate(parts):
        L = len(idxs)
        idx_tensor[p, :L] = torch.tensor(idxs, dtype=torch.long, device=device)
        mask_tensor[p, :L] = True
    return idx_tensor, mask_tensor  # shapes: (P, max_len)

# ------------------------------------------------------------
# 1) PoseNormalizer
# ------------------------------------------------------------
class PoseNormalizer(nn.Module):
    """
    Normalise per-frame joint coords:
    - Center at pelvis (hip midpoint)
    - Scale by shoulder-hip distance (fallback: shoulder width)
    Confidences are preserved; provides a visibility mask.
    Assumes COCO-like indexing; tweak indices if needed.
    """
    def __init__(self):
        super(PoseNormalizer, self).__init__()
        # COCO/ViTPose-17 default indices
        self.hip_l, self.hip_r = 11, 12
        self.sh_l,  self.sh_r  = 5,  6

    def forward(self, x):
        """
        x: (B,T,J,3) with (x,y,conf)
        returns: x_norm (B,T,J,3) and mask (B,T,J,1)
        """
        coords = x[..., :2]        # (B,T,J,2)
        conf   = x[..., 2:3]       # (B,T,J,1)
        mask   = (conf > 0.05).float()

        # center at pelvis (hips midpoint)
        pelvis = (coords[..., self.hip_l, :] + coords[..., self.hip_r, :]) / 2.0  # (B,T,2)
        pelvis = pelvis.unsqueeze(2)  # (B,T,1,2)
        coords_centered = coords - pelvis

        # scale by shoulder-hip distance; fallback: shoulder width
        vec = coords[..., self.sh_l, :] - coords[..., self.hip_l, :]
        scale = torch.norm(vec, dim=-1, keepdim=True)  # (B,T,1)
        fallback = torch.norm(coords[..., self.sh_l, :] - coords[..., self.sh_r, :],
                              dim=-1, keepdim=True) + 1e-6
        scale = torch.where(scale < 1e-3, fallback, scale)
        scale = scale.unsqueeze(2)  # (B,T,1,1)

        coords_norm = coords_centered / (scale + 1e-6)
        x_norm = torch.cat([coords_norm, conf], dim=-1)
        return x_norm, mask

# ------------------------------------------------------------
# 2) Per-Joint Encoder (MLP)
# ------------------------------------------------------------
class JointEncoder(nn.Module):
    def __init__(self, in_dim=3, hidden=128, out_dim=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, mask):
        """
        x:    (B,T,J,3)
        mask: (B,T,J,1) in {0,1}
        returns: (B,T,J,D)
        """
        B,T,J,_ = x.shape
        h = self.mlp(x.view(B*T*J, -1)).view(B,T,J,-1)
        if mask is not None:
            h = h * mask
        return h

# ------------------------------------------------------------
# 3) Body-Part Attention (per-frame attention over parts)
# ------------------------------------------------------------
class BodyPartAttention(nn.Module):
    """
    Steps:
      1) Gather joint features per body part
      2) Mask-aware mean pool -> part tokens (B,T,P,D)
      3) Multi-head self-attention across parts (per frame)
      4) Residual + LayerNorm
    Returns per-part importance (B,T,P).
    """
    def __init__(self, body_parts: dict[str, list[int]], D: int, num_heads=4, dropout=0.1):
        super(BodyPartAttention, self).__init__()
        self.body_parts = body_parts
        self.part_names = list(body_parts.keys())
        self.P = len(self.part_names)
        self.proj_in  = nn.Linear(D, D)
        self.attn     = nn.MultiheadAttention(embed_dim=D, num_heads=num_heads,
                                              dropout=dropout, batch_first=True)
        self.ln       = nn.LayerNorm(D)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, h_joint, mask_joint):
        """
        h_joint:    (B,T,J,D)
        mask_joint: (B,T,J,1)  float {0,1} from upstream; we’ll convert to bool for masking
        """
        device = h_joint.device
        B, T, J, D = h_joint.shape

        # 1) Build part indices (P, L) with -1 padded entries
        idx_tensor, idx_mask = build_part_index_tensor(self.body_parts, J, device)  # (P,L) long, (P,L) bool
        P, L = idx_tensor.shape

        # --- Sanity check: any real index out of bounds? ---
        bad_hi = (idx_tensor >= 0) & (idx_tensor >= J)
        bad_lo = (idx_tensor < -1)
        if bad_hi.any() or bad_lo.any():
            # Optional: raise with details so you can fix the body_parts dict
            bad_positions = torch.nonzero(bad_hi | bad_lo, as_tuple=False)
            raise ValueError(f"Body-part indices out of range for J={J}. Offenders at {bad_positions.tolist()}.")

        # 2) Make a VALIDITY mask for non-padded entries, and a SAFE index tensor (replace -1 -> 0)
        valid_mask = (idx_tensor >= 0)  # (P, L) bool: True means real joint index
        safe_idx = idx_tensor.clamp(min=0)  # (P, L) long: -1 becomes 0 (dummy)

        # 3) Shape indices to match gather/take_along_dim on dim=3 (joint dim)
        #    We want output (B,T,P,L,D), so index needs shape (B,T,P,L,D)
        idx_full = safe_idx.view(1, 1, P, L, 1).expand(B, T, P, L, D)  # long
        valid_full = valid_mask.view(1, 1, P, L, 1).expand(B, T, P, L, 1)  # bool

        # 4) Gather features and visibility masks
        h_exp = h_joint.unsqueeze(2)  # (B,T,1,J,D)
        m_exp = (mask_joint > 0.05).unsqueeze(2)  # (B,T,1,J,1)  bool

        h_gathered = torch.take_along_dim(h_exp, idx_full, dim=3)  # (B,T,P,L,D)
        m_gathered = torch.take_along_dim(m_exp, idx_full[..., :1], dim=3)  # (B,T,P, L,1) bool

        # 5) Invalidate padded positions (where original index was -1)
        m_gathered = m_gathered & valid_full  # keep only real joints
        m_gathered_f = m_gathered.float()  # to weights

        # 6) Mask-aware mean over L joints -> (B,T,P,D)
        eps = 1e-6
        denom = m_gathered_f.sum(dim=3).clamp_min(eps)  # (B,T,P,1)
        part_tokens = (h_gathered * m_gathered_f).sum(dim=3) / denom  # (B,T,P,D)

        # Linear proj + self-attn over parts (per frame)
        part_tokens = self.proj_in(part_tokens)  # (B,T,P,D)
        bt = B * T
        q = part_tokens.reshape(bt, P, D)  # (B*T, P, D)
        attn_out, attn_weights = self.attn(q, q, q, need_weights=True, average_attn_weights=True)
        attn_out = self.dropout(attn_out)
        attn_out = self.ln(q + attn_out)
        part_out = attn_out.view(B, T, P, D)

        # Part importance: average attention received per part
        # attn_weights: (num_heads, B*T, P, P)
        bt = B * T
        assert attn_weights.shape == (bt, P, P), f"got {attn_weights.shape}, expected {(bt, P, P)}"

        # average over queries to get how much attention each part receives
        part_importance = attn_weights.mean(dim=1)  # (B*T, P)
        part_importance = part_importance.view(B, T, P)

        return part_out, part_importance


# ------------------------------------------------------------
# 4) Temporal Transformer
# ------------------------------------------------------------
class TemporalTransformer(nn.Module):
    def __init__(self, in_dim, n_layers=2, n_heads=4, ff=512, dropout=0.1, cls_token=True):
        super(TemporalTransformer,self).__init__()
        self.cls_token = cls_token
        self.token = nn.Parameter(torch.zeros(1,1,in_dim)) if cls_token else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, x):
        """
        x: (B,T,D) sequence
        returns: pooled (B,D), sequence (B,T,D)
        """
        if self.cls_token is not None:
            B = x.size(0)
            cls = self.token.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)  # (B, T+1, D)
        y = self.encoder(x)
        y = self.ln(y)
        if self.cls_token is not None:
            pooled = y[:, 0]   # (B,D)
            seq    = y[:, 1:]  # (B,T,D)
        else:
            pooled = y.mean(dim=1)
            seq    = y
        return pooled, seq

# ------------------------------------------------------------
# 5) PoseBodyPartAttentionModel (Backbone)
# ------------------------------------------------------------
class PoseBodyPartAttentionModel(nn.Module):
    """
    Backbone that turns a sequence of 2D poses into:
      - Embedding z (B,E) for similarity
      - Part attention (B,T,P) for interpretability
      - (Optional) logits (B,C) if classifier enabled

    Input: x (B,T,J,3) = (x,y,conf) per joint
    """
    def __init__(
        self,
        body_parts: dict[str, list[int]],
        joint_in_dim: int = 3,          # (x,y,conf) or add velocities -> 5
        joint_hidden: int = 128,
        joint_out_dim: int = 128,
        part_heads: int = 4,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        emb_dim: int = 128,
        num_classes: int | None = None,
        dropout: float = 0.1,
    ):
        super(PoseBodyPartAttentionModel, self).__init__()
        self.body_parts = body_parts
        self.normalizer   = PoseNormalizer()
        self.joint_enc    = JointEncoder(joint_in_dim, joint_hidden, joint_out_dim, dropout)
        self.part_attn    = BodyPartAttention(body_parts, D=joint_out_dim, num_heads=part_heads, dropout=dropout)
        self.temporal_in  = nn.Linear(joint_out_dim * len(body_parts), joint_out_dim)
        self.temporal     = TemporalTransformer(in_dim=joint_out_dim, n_layers=temporal_layers,
                                                n_heads=temporal_heads, dropout=dropout)
        self.emb_head     = nn.Sequential(nn.Linear(joint_out_dim, emb_dim), nn.LayerNorm(emb_dim))
        self.cls_head     = nn.Linear(joint_out_dim, num_classes) if num_classes is not None else None

    def forward(self, x):
        """
        x: (B,T,J,3) with (x,y,conf)
        returns:
          z: (B,E) L2-normalised embedding
          part_attn: (B,T,P) per-part importance over time
          logits: (B,num_classes) or None
        """
        x_norm, vis_mask = self.normalizer(x)                  # (B,T,J,3), (B,T,J,1)
        h_joint = self.joint_enc(x_norm, vis_mask)             # (B,T,J,D)
        h_part, part_attn = self.part_attn(h_joint, vis_mask)  # (B,T,P,D), (B,T,P)

        # Flatten parts per frame -> temporal sequence
        B,T,P,D = h_part.shape
        h_frame = h_part.reshape(B, T, P*D)                    # (B,T,P*D)
        h_frame = self.temporal_in(h_frame)                    # (B,T,D)

        pooled, _ = self.temporal(h_frame)                     # (B,D), (B,T,D)
        z = F.normalize(self.emb_head(pooled), dim=-1)         # (B,E)

        logits = self.cls_head(pooled) if self.cls_head is not None else None
        return z, part_attn, logits
