from models.linear.siamese_trainable import SiameseModelTrainable
from models.pose_attn.body_parts import VITPOSE_COCO17
from models.pose_attn.pose_bodypart_attn import PoseBodyPartAttentionModel
from models.pose_attn.siamese_wrapper import SiameseWrapper
from models.resnet.resnet_siamese import ResNetSiamese
from models.r3d18.r3d_siamese import R3D18Siamese

def create_model(backbone_kind: str, **kwargs):
    backbone_kind = backbone_kind.lower()
    if backbone_kind == "pure":
        return SiameseModelTrainable(**kwargs)
    if backbone_kind == "resnet18":
        return ResNetSiamese(**kwargs)
    if backbone_kind == "r3d_18":
        return R3D18Siamese(**kwargs)
    if backbone_kind == "pose_attn":
        body_parts = kwargs.pop("body_parts", VITPOSE_COCO17)
        margin     = kwargs.pop("margin", 0.5)
        return_attn = kwargs.pop("return_attn", True)
        backbone = PoseBodyPartAttentionModel(body_parts=body_parts, **kwargs)
        return SiameseWrapper(backbone, margin=margin, return_attn=return_attn)
    raise ValueError("Unknown backbone_kind: {}".format(backbone_kind))
