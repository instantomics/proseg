from __future__ import annotations

from .method import load_config, posterior_method


def posterior(inputs, outputs):
    posterior_method(inputs, outputs)


__all__ = ["load_config", "posterior"]
