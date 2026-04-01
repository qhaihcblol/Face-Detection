from collections import OrderedDict

from torch import nn, Tensor

def _make_divisible(v: float, divisor: int = 8) -> int:
    """This function ensures that all layers have a channel number divisible by 8"""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class IntermediateLayerGetterByIndex(nn.Module):
    def __init__(self, model: nn.Module, indexes: list[int] | None = None) -> None:
        super().__init__()
        features = getattr(model, "features", None)
        if not isinstance(features, nn.Sequential):
            raise TypeError("model.features must be an nn.Sequential")
        self.features: nn.Sequential = features
        self.indexes = tuple(indexes or [6, 13, 18])

    def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
        outputs: OrderedDict[str, Tensor] = OrderedDict()
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.indexes:
                out_name = f"layer_{idx}"
                outputs[out_name] = x

        return outputs
