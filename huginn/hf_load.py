"""Load tomg-group-umd/huginn-0125 under transformers 5.x.

The checkpoint's remote code was written for transformers 4.44: it declares `_tied_weights_keys` as a list,
which 5.x expects as a {target: source} dict. Patch that one attribute before instantiating.
"""

import torch
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

REPO = "tomg-group-umd/huginn-0125"


def load_huginn(device="cuda", dtype=torch.bfloat16):
    cls = get_class_from_dynamic_module("raven_modeling_minimal.RavenForCausalLM", REPO)
    for klass in cls.__mro__:
        if "_tied_weights_keys" in vars(klass):
            klass._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
    model = cls.from_pretrained(REPO, dtype=dtype).to(device).eval()
    return model, AutoTokenizer.from_pretrained(REPO)
