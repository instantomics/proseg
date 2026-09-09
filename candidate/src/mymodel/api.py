from __future__ import annotations

from .method import posterior_method


def posterior(inputs, outputs):
    posterior_method(inputs, outputs)


__all__ = ["posterior"]
