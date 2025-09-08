from typing import List

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x: List[List[float]], training=True) -> List[List[float]]:
        out = x
        for layer in self.layers:
            # forward with (training) if available
            try:
                out = layer(out, training=training)
            except TypeError:
                out = layer(out)
        return out
