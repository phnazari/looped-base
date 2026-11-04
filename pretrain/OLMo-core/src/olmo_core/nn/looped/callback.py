"""Trainer callback of the looped models: logs the learned depth prior and checkpoints the training
state the weights do not hold (the forward index that seeds the depth draws, and the prior)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from olmo_core.train.callbacks.callback import Callback


@dataclass
class LoopedDepthCallback(Callback):
    """Pairs with :class:`~olmo_core.nn.looped.model.LoopedTransformer`. The model updates the prior
    in its ``post_optim_step``; this callback records the step's prior metrics and saves and restores
    the depth state with the trainer state."""

    def _model(self):
        from .model import LoopedTransformer

        model = self.trainer.train_module.model
        if not isinstance(model, LoopedTransformer):
            raise TypeError("LoopedDepthCallback needs a LoopedTransformer")
        return model

    def post_train_batch(self):
        model = self._model()
        for name, value in model.depth_prior_metrics.items():
            self.trainer.record_metric(name, value)
        model.depth_prior_metrics = {}

    def state_dict(self) -> Dict[str, Any]:
        return {"looped": self._model().training_state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._model().load_training_state_dict(state_dict["looped"])
