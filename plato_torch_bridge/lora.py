"""Style LoRA Adapter — conditions a language model on a style embedding.

This is LoRA applied to style conditioning, not to the LM weights themselves.
Architecture:
    Linear(32, r) → Linear(r, d_model)
where r << d_model is the low-rank bottleneck.

This adapter can be fine-tuned per composer while keeping the base model frozen.
It projects a 32-dim style embedding into the LM's embedding space for
conditioning token generation.
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object  # type: ignore


class StyleLoRAAdapter(nn.Module):
    """Maps 32-dim style embedding → LM-compatible conditioning vector.

    Architecture:
        Linear(32, r) → ReLU → Linear(r, d_model)

    Where r << d_model is the low-rank bottleneck. This is LoRA applied to the
    style conditioning path, not to the LM weights themselves.

    The adapter can be fine-tuned per composer while keeping the contrastive
    encoder frozen — only the r × d_model parameters need to be stored/sent.

    Input:  (B, 32) style embeddings (from ContrastiveStyleEncoder)
    Output: (B, d_model) conditioning vectors ready for LM injection
    """

    def __init__(self, embedding_dim: int = 32,
                 rank: int = 4,
                 d_model: int = 4096):
        """Initialize the LoRA adapter.

        Args:
            embedding_dim: Input style embedding dimension (default 32)
            rank: Low-rank bottleneck size (default 4, r << d_model)
            d_model: Output LM embedding dimension (default 4096)
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required. Install with: pip install plato-midi-bridge[torch]"
            )
        super().__init__()

        self.embedding_dim = embedding_dim
        self.rank = rank
        self.d_model = d_model

        # Low-rank decomposition
        self.down = nn.Linear(embedding_dim, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

        # Optional: learnable scaling factor (like LoRA's alpha/r)
        self.scale = nn.Parameter(torch.ones(1))

        # Initialize: down is small (like Kaiming), up is zero
        # This ensures the adapter starts as zero output
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, style_embedding: torch.Tensor) -> torch.Tensor:
        """Project style embedding to LM conditioning space.

        Args:
            style_embedding: (B, 32) L2-normalized style embeddings

        Returns:
            (B, d_model) conditioning vectors
        """
        # Down-project to low-rank space
        bottleneck = F.relu(self.down(style_embedding))
        # Up-project to LM space
        output = self.up(bottleneck)
        # Apply learned scale
        output = output * self.scale
        return output

    def get_lora_parameters(self) -> int:
        """Count the number of LoRA parameters (r * (32 + d_model)).

        Returns:
            Total trainable parameter count for this adapter.
        """
        return (self.embedding_dim * self.rank) + (self.rank * self.d_model)

    def merge_with(self, other: "StyleLoRAAdapter",
                   alpha: float = 0.5) -> "StyleLoRAAdapter":
        """Merge this adapter with another by linear interpolation.

        Useful for interpolating between composer styles:
            merged = alpha * self + (1 - alpha) * other

        Args:
            other: Another StyleLoRAAdapter with same architecture
            alpha: Interpolation weight for this adapter (default 0.5)

        Returns:
            New StyleLoRAAdapter with merged weights
        """
        merged = StyleLoRAAdapter(
            embedding_dim=self.embedding_dim,
            rank=self.rank,
            d_model=self.d_model
        )
        with torch.no_grad():
            merged.down.weight.data = (
                alpha * self.down.weight.data +
                (1.0 - alpha) * other.down.weight.data
            )
            merged.up.weight.data = (
                alpha * self.up.weight.data +
                (1.0 - alpha) * other.up.weight.data
            )
            merged.scale.data = (
                alpha * self.scale.data +
                (1.0 - alpha) * other.scale.data
            )
        return merged

    def selftest(self) -> bool:
        """Run a quick forward pass to verify dimensions.

        Returns:
            True if forward pass produces correct output shape.
        """
        with torch.no_grad():
            x = torch.randn(2, self.embedding_dim)
            out = self.forward(x)
            expected = (2, self.d_model)
            return out.shape == expected



