"""Configuration of the looped models of *Towards Looped Models Done Right* (Parts I and II).

The models are ported from the official release, xLLM-Loop (https://github.com/ifm-ai/xllm-loop,
Apache-2.0), ``xllm/models/looped.py``. Each field here maps to one field of that release's
``ModelConf``; the docstrings name it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

ARCHITECTURES = ("looped_transformer", "huginn")
STATE_INITS = ("zero", "input", "normal", "truncated_normal")
HIERARCHICAL_STATES = ("none", "shared_hl", "split_hl")
SAMPLING_SCHEMES = ("fixed", "poisson_lognormal")


@dataclass
class LoopedConfig(Config):
    """
    ``looped_transformer`` (xLLM ``arch='transformer'``, the Ouro-style loop): blocks
    ``[loop_start_layer, loop_end_layer)`` run ``loop_times`` times, optionally re-injecting the
    embeddings at every loop entry.

    ``huginn`` (xLLM ``arch='huginn'``): a prelude, a recurrent block iterated ``loop_times`` times
    from an initial state with the prelude output injected at every recurrence, and a coda. The
    recurrent state can be split into hierarchical H/L states (Part I). With ``depth_control``
    (xLLM ``huginn_depth_control``, Part II) the training depth is fixed, sampled or learned, the
    first recurrences beyond ``backprop_depth`` run without gradients, and the injection variants
    of Part II apply.

    """

    architecture: str
    loop_times: int = 1  # loop_times

    # looped_transformer
    loop_start_layer: int = 0  # loop_start_layer
    loop_end_layer: Optional[int] = None  # loop_end_layers; None means n_layers
    loop_input_injection: str = "none"  # loop_input_injection: none or diagonal
    loop_diagonal_dt_init: float = 1.0  # loop_diagonal_dt_init
    loop_diagonal_a_init: float = 1.0  # loop_diagonal_a_init
    # The unrolled control (loop_times 1): the output of block `prelude_source_layer` is injected
    # before each block of `prelude_input_layers` (dense_prelude_*).
    prelude_input_injection: str = "none"  # dense_prelude_input_injection: none or diagonal
    prelude_input_layers: tuple[int, ...] = ()  # dense_prelude_input_layers
    prelude_source_layer: int = -1  # dense_prelude_source_layer
    prelude_diagonal_dt_init: float = 1.0  # dense_prelude_diagonal_dt_init
    prelude_diagonal_a_init: float = 1.0  # dense_prelude_diagonal_a_init

    # huginn
    prelude_layers: int = 1  # huginn_prelude_layers
    recurrent_layers: Optional[int] = None  # huginn_recurrent_layers; None means the rest
    coda_layers: int = 1  # huginn_coda_layers
    state_init: str = "zero"  # huginn_state_init: zero, input or normal
    input_injection: str = "diagonal"  # huginn_input_injection: none, diagonal or linear
    diagonal_dt_init: float = 1.0  # huginn_diagonal_dt_init
    diagonal_a_init: float = 1.0  # huginn_diagonal_a_init
    hierarchical_state: str = "none"  # huginn_hierarchical_state: none, shared_hl or split_hl
    hierarchical_h_cycles: int = 2  # huginn_hierarchical_h_cycles
    hierarchical_l_cycles: int = 3  # huginn_hierarchical_l_cycles
    split_module_repeats: int = 1  # huginn_split_module_repeats

    # huginn with depth control (Part II)
    depth_control: bool = False  # huginn_depth_control
    sampling: str = "fixed"  # huginn_sampling_scheme: fixed or poisson-lognormal-capped
    antithetic_sampling: bool = False  # huginn_antithetic_sampling
    poisson_lognormal_mean: float = 5.0  # huginn_poisson_lognormal_target_mean
    poisson_lognormal_sigma: float = 0.5  # huginn_poisson_lognormal_sigma
    poisson_lognormal_max: int = 64  # huginn_poisson_lognormal_max
    backprop_depth: Optional[int] = None  # huginn_backprop_depth; None backpropagates all
    learned_depth_prior: bool = False  # huginn_depth_prior == "learned"
    depth_prior_entropy: float = 0.0  # huginn_depth_prior_entropy (lambda_H)
    prelude_norm: bool = False  # huginn_prelude_norm == "rms" (Parcae-Decay)
    recurrent_exit_norm: bool = False  # huginn_recurrent_exit_norm == "parameterless_rms"
    orthogonal_injection: bool = False  # huginn_prelude_orthogonal (OrthoInj)
    orthogonal_eps: float = 1e-6  # huginn_ortho_projection_eps
    linear_injection_bias: bool = True  # apply_bias_term, which only the linear injection reads

    # Not in the release: what the released Huginn-0125 checkpoint (Geiping et al. 2025) needs.
    # ``truncated_normal`` draws the initial state from N(0, state_init_std^2) truncated at 3 std,
    # times the embedding scale, in training and evaluation alike. ``qk_bias`` adds biases to the
    # query and key projections only. ``recurrence_final_norm`` applies the LM head's norm once to
    # the last recurrent state (Huginn's ``ln_f`` serves both places).
    state_init_std: float = 0.0
    qk_bias: bool = False
    recurrence_final_norm: bool = False

    def validate(self, n_layers: int) -> None:
        def fail(message: str):
            raise OLMoConfigurationError(f"looped: {message}")

        if self.architecture not in ARCHITECTURES:
            fail(f"architecture must be one of {ARCHITECTURES}")
        if self.loop_times < 1:
            fail("loop_times must be positive")
        if self.architecture == "looped_transformer":
            end = self.end_layer(n_layers)
            if not 0 <= self.loop_start_layer < end <= n_layers:
                fail("invalid loop layer span")
            if self.loop_input_injection not in ("none", "diagonal"):
                fail("loop_input_injection must be none or diagonal")
            if self.prelude_input_injection not in ("none", "diagonal"):
                fail("prelude_input_injection must be none or diagonal")
            if self.prelude_input_injection == "none":
                if self.prelude_input_layers or self.prelude_source_layer >= 0:
                    fail("prelude_input_layers and prelude_source_layer need prelude_input_injection")
            else:
                if self.loop_input_injection != "none" or self.loop_times != 1:
                    fail("the unrolled prelude injection needs loop_times 1 and no loop injection")
                if not self.prelude_input_layers:
                    fail("prelude_input_layers is empty")
                if not 0 <= self.prelude_source_layer < n_layers:
                    fail("prelude_source_layer is outside the model")
                if any(not self.prelude_source_layer < layer < n_layers for layer in self.prelude_input_layers):
                    fail("every prelude input layer must come after the source layer")
            if self.depth_control:
                fail("depth_control applies to huginn only")
            if self.state_init_std > 0 or self.recurrence_final_norm:
                fail("state_init_std and recurrence_final_norm apply to huginn only")
            return

        recurrent = self.recurrent_count(n_layers)
        if self.prelude_layers < 0 or self.coda_layers < 0 or recurrent < 1:
            fail("prelude, recurrent and coda layer counts are invalid")
        if self.prelude_layers + recurrent + self.coda_layers != n_layers:
            fail(f"prelude + recurrent + coda layers must equal n_layers={n_layers}")
        if self.state_init not in STATE_INITS:
            fail(f"state_init must be one of {STATE_INITS}")
        if (self.state_init == "truncated_normal") != (self.state_init_std > 0):
            fail("state_init_std is the scale of state_init=truncated_normal, and only of it")
        if self.recurrence_final_norm and self.hierarchical_state != "none":
            fail("recurrence_final_norm supports a single recurrent state")
        if self.input_injection not in ("none", "diagonal", "linear"):
            fail("input_injection must be none, diagonal or linear")
        if self.hierarchical_state not in HIERARCHICAL_STATES:
            fail(f"hierarchical_state must be one of {HIERARCHICAL_STATES}")
        if self.hierarchical_state != "none":
            if self.hierarchical_h_cycles < 1 or self.hierarchical_l_cycles < 1:
                fail("hierarchical cycles must be positive")
            if self.split_module_repeats < 1:
                fail("split_module_repeats must be positive")
            if self.hierarchical_state == "split_hl" and recurrent < 2:
                fail("split_hl needs at least two recurrent layers")
        elif self.split_module_repeats != 1:
            fail("split_module_repeats applies to hierarchical states only")
        part_two = (self.sampling != "fixed" or self.backprop_depth is not None or self.learned_depth_prior
                    or self.prelude_norm or self.recurrent_exit_norm or self.orthogonal_injection
                    or self.input_injection == "linear")
        if part_two and not self.depth_control:
            fail("sampled depths, truncation, the learned prior and the Part II injections need depth_control")
        if not self.depth_control:
            return
        if self.hierarchical_state != "none":
            fail("depth_control supports a single recurrent state")
        if self.sampling not in SAMPLING_SCHEMES:
            fail(f"sampling must be one of {SAMPLING_SCHEMES}")
        if self.backprop_depth is not None and self.backprop_depth < 1:
            fail("backprop_depth must be positive")
        if self.learned_depth_prior and self.sampling != "poisson_lognormal":
            fail("the learned depth prior starts from the Poisson-lognormal prior")
        if self.orthogonal_injection and self.input_injection != "diagonal":
            fail("the orthogonal injection is a diagonal injection")
        if self.sampling == "poisson_lognormal":
            if self.poisson_lognormal_max < 1 or self.poisson_lognormal_sigma <= 0:
                fail("invalid Poisson-lognormal parameters")
            if self.learned_depth_prior and self.poisson_lognormal_max != 64:
                fail("the learned depth prior has support 1..64")

    def end_layer(self, n_layers: int) -> int:
        return n_layers if self.loop_end_layer is None else self.loop_end_layer

    def recurrent_count(self, n_layers: int) -> int:
        if self.recurrent_layers is not None:
            return self.recurrent_layers
        return n_layers - self.prelude_layers - self.coda_layers

    def num_params(self, d_model: int) -> int:
        """Parameters outside the transformer blocks: injections and the prelude norm. The query and
        key biases of ``qk_bias`` sit inside the blocks and are not counted here."""
        diagonal = d_model * d_model + 2 * d_model
        if self.architecture == "looped_transformer":
            return diagonal * ((self.loop_input_injection != "none") + (self.prelude_input_injection != "none"))
        params = 0
        if self.input_injection == "diagonal":
            params += diagonal
        elif self.input_injection == "linear":
            params += 2 * d_model * d_model + (d_model if self.linear_injection_bias else 0)
        if self.prelude_norm:
            params += d_model
        return params
