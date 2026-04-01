from .backbones.mobilenet_v2 import mobilenet_v2
from .utils import IntermediateLayerGetterByIndex

BACKBONE_FACTORY = {
    "mobilenet_v2": mobilenet_v2,
}


def build_backbone(name, pretrained=False):
    if name not in BACKBONE_FACTORY:
        raise ValueError(f"Unknown backbone: {name}")

    return BACKBONE_FACTORY[name](pretrained=pretrained)


def get_layer_extractor(config, backbone):
    if config["name"] == "mobilenet_v2":
        return IntermediateLayerGetterByIndex(
            backbone, indexes=config["return_layers"]
        )

    raise ValueError(f"Unsupported layer extractor for backbone: {config['name']}")
