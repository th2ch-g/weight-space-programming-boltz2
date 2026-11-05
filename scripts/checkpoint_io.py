"""Load tensors and the fixed OmegaConf metadata used by Boltz checkpoints."""

from collections import defaultdict
from importlib.metadata import version
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from packaging.version import Version

_CHECKPOINT_GLOBALS = (
    DictConfig,
    ListConfig,
    Metadata,
    ContainerMetadata,
    AnyNode,
    Any,
    defaultdict,
    dict,
    list,
    int,
)


def load_checkpoint(path, *, map_location="cpu", mmap=True):
    if Version(torch.__version__) < Version("2.13.0"):
        raise RuntimeError("Checkpoint loading requires torch>=2.13.0; sync the current environment")
    with torch.serialization.safe_globals(_CHECKPOINT_GLOBALS):
        return torch.load(path, map_location=map_location, mmap=mmap, weights_only=True)


def load_model_checkpoint(model_type, *args, **kwargs):
    if kwargs.get("weights_only", True) is not True:
        raise ValueError("Unrestricted checkpoint loading is not supported")
    if Version(torch.__version__) < Version("2.13.0"):
        raise RuntimeError("Checkpoint loading requires torch>=2.13.0; sync the current environment")
    if Version(version("pytorch-lightning")) < Version("2.6.6"):
        raise RuntimeError("Checkpoint loading requires pytorch-lightning>=2.6.6")
    kwargs["weights_only"] = True
    with torch.serialization.safe_globals(_CHECKPOINT_GLOBALS):
        return model_type.load_from_checkpoint(*args, **kwargs)
