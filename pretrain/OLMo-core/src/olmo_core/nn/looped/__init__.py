"""Looped models of *Towards Looped Models Done Right* (Parts I and II), ported from xLLM-Loop.

The model is :class:`olmo_core.nn.looped.model.LoopedTransformer`; a
:class:`~olmo_core.nn.transformer.TransformerConfig` builds it when its ``looped`` field is set.
"""

from .config import LoopedConfig

__all__ = ["LoopedConfig"]
