# models/__init__.py
from models.linear.siamese_trainable import SiameseModelTrainable
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
    raise ValueError("Unknown backbone_kind: {}".format(backbone_kind))
