import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """A minimal subclass of `torch.nn.Module` that implements common utilities."""

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):  # noqa: D401,E501 pylint: disable=unused-argument
        raise NotImplementedError("Subclasses must implement forward()") 