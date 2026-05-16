"""plato_torch_bridge — PyTorch bridge for plato-midi-bridge style decomposition.

This module bridges the plato-midi-bridge style decomposer with PyTorch.
It provides datasets, encoders, and LoRA adapters for training on MIDI style vectors.

All torch imports are gated so the main decomposer doesn't depend on torch.
Use `pip install plato-midi-bridge[torch]` for this extra.
"""

from .dataset import MIDIStyleDataset
from .encoder import ContrastiveStyleEncoder
from .lora import StyleLoRAAdapter

__all__ = [
    "MIDIStyleDataset",
    "ContrastiveStyleEncoder",
    "StyleLoRAAdapter",
]
