"""
EMA for any PyTorch model.

Carried over from ``any_nn.AnyEMA`` unchanged, so a trainer that used it keeps
working.  It lives on the training machine and never crosses the wire; what
crosses is whatever ``save_checkpoint`` decides to hand back, which for an EMA
run is usually the shadow weights rather than the live ones.
"""

from __future__ import annotations

import torch


class AnyEMA:
    def __init__(self, named_parameters, scale: float = 1.0, capture=()):
        capture = tuple(capture)
        self.ema_parameters: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in named_parameters:
                if param.requires_grad and (not capture or name in capture):
                    self.ema_parameters[name] = param.data.clone().detach() * scale

    @torch.no_grad()
    def update(self, named_parameters, decay: float = 0.999):
        for name, param in named_parameters:
            shadow = self.ema_parameters.get(name)
            if shadow is not None and param.requires_grad:
                shadow.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @torch.no_grad()
    def swap(self, named_parameters):
        """Exchange live weights with the shadow.  Call twice to restore."""
        for name, param in named_parameters:
            shadow = self.ema_parameters.get(name)
            if shadow is not None and param.requires_grad:
                live = param.data.clone()
                param.data.copy_(shadow)
                shadow.copy_(live)
