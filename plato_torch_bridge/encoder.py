"""Contrastive Style Encoder — maps 109-dim style vectors to 32-dim latent embeddings.

Uses triplet loss (or NT-Xent / SimCLR-style) to attract same-composer embeddings
and repel different-composer embeddings. Architecture is a simple 3-layer MLP.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object  # type: ignore


class ContrastiveStyleEncoder(nn.Module):
    """Maps 109-dim style vectors → 32-dim latent style embeddings.

    Architecture:
        Linear(109, 64) → BatchNorm → ReLU → Dropout(0.1)
        Linear(64, 48) → BatchNorm → ReLU → Dropout(0.1)
        Linear(48, 32) → L2-normalize

    Same-composer positive pairs are attracted via triplet/margin loss,
    while different-composer pairs are repelled.

    The output is L2-normalized so cosine similarity can be used directly.

    Input:  (B, 109) style vectors
    Output: (B, 32) L2-normalized embeddings
    """

    def __init__(self, input_dim: int = 109,
                 hidden_dim_1: int = 64,
                 hidden_dim_2: int = 48,
                 embedding_dim: int = 32,
                 dropout: float = 0.1):
        """Initialize the encoder.

        Args:
            input_dim: Input style vector dimension (default 109)
            hidden_dim_1: First hidden layer size (default 64)
            hidden_dim_2: Second hidden layer size (default 48)
            embedding_dim: Output embedding dimension (default 32)
            dropout: Dropout probability (default 0.1)
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required. Install with: pip install plato-midi-bridge[torch]"
            )
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim_2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode style vectors to L2-normalized embeddings.

        Args:
            x: Input tensor of shape (B, input_dim) — 109-dim style vectors

        Returns:
            L2-normalized embeddings of shape (B, embedding_dim)
        """
        embedding = self.net(x)
        # L2 normalize along the embedding dimension
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    # ── Loss Functions ──────────────────────────────────

    def triplet_loss(self, anchor: torch.Tensor,
                     positive: torch.Tensor,
                     negative: torch.Tensor,
                     margin: float = 0.5) -> torch.Tensor:
        """Triplet loss: max(0, ||a-p||² - ||a-n||² + margin).

        Anchors, positives, and negatives are already L2-normalized embeddings.

        Args:
            anchor: Anchor embeddings (B, embedding_dim)
            positive: Same-composer positive embeddings (B, embedding_dim)
            negative: Different-composer negative embeddings (B, embedding_dim)
            margin: Minimum margin between pos/neg distances (default 0.5)

        Returns:
            Scalar triplet loss
        """
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = F.relu(pos_dist - neg_dist + margin)
        return loss.mean()

    def contrastive_loss(self, anchor: torch.Tensor,
                         positive: torch.Tensor,
                         negative: torch.Tensor,
                         margin: float = 0.5) -> torch.Tensor:
        """Alias for triplet_loss."""
        return self.triplet_loss(anchor, positive, negative, margin)

    def nt_xent_loss(self, embeddings: torch.Tensor,
                     labels: torch.Tensor,
                     temperature: float = 0.1) -> torch.Tensor:
        """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        SimCLR-style contrastive loss. For each sample, all other samples
        with the same label are positive pairs; different labels are negatives.

        Args:
            embeddings: L2-normalized embeddings (B, embedding_dim)
            labels: Class labels (B,) — samples with same label are positive pairs
            temperature: Temperature scaling (default 0.1)

        Returns:
            Scalar NT-Xent loss
        """
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, requires_grad=True)

        # Compute cosine similarity matrix (embeddings already L2-normalized)
        sim = embeddings @ embeddings.T  # (B, B)

        # Scale by temperature
        sim = sim / temperature

        # Create mask: positive pairs = same label, exclude self
        labels_expanded = labels.unsqueeze(0)  # (1, B)
        labels_expanded_t = labels.unsqueeze(1)  # (B, 1)
        pos_mask = (labels_expanded == labels_expanded_t).float()  # (B, B)
        # Remove self-pairs
        self_mask = torch.eye(B, device=embeddings.device, dtype=torch.float32)
        pos_mask = pos_mask - self_mask
        # Negative mask = everything except positive
        neg_mask = 1.0 - pos_mask - self_mask

        # For each anchor, compute loss over all pairs
        # Numerator: exp(sim) for positive pairs
        # Denominator: sum(exp(sim)) for all pairs (including self)
        exp_sim = torch.exp(sim)

        pos_sum = (exp_sim * pos_mask).sum(dim=1)  # (B,)
        total_sum = exp_sim.sum(dim=1)  # (B,)

        # Avoid division by zero: if no positive pairs, loss = 0 for that sample
        loss_per_sample = -torch.log(
            pos_sum / (total_sum + 1e-8) + 1e-8
        )
        loss_per_sample = loss_per_sample * (pos_mask.sum(dim=1) > 0).float()

        return loss_per_sample.mean()

    # ── Convenience Methods ─────────────────────────────

    def encode_style_vector(self, style_vector: np.ndarray) -> np.ndarray:
        """Encode a single style vector to its embedding.

        Args:
            style_vector: (109,) numpy style vector

        Returns:
            (32,) numpy embedding
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(style_vector).float().unsqueeze(0)
            embedding = self.forward(tensor)
            return embedding.squeeze(0).numpy()

    def encode_batch(self, style_vectors: np.ndarray) -> np.ndarray:
        """Encode a batch of style vectors.

        Args:
            style_vectors: (N, 109) numpy array

        Returns:
            (N, 32) numpy array of embeddings
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(style_vectors).float()
            embeddings = self.forward(tensor)
            return embeddings.numpy()

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two style vector embeddings.

        Args:
            a: (109,) or (32,) style vector or embedding
            b: (109,) or (32,) style vector or embedding

        Returns:
            Cosine similarity in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            if a.shape[-1] == 109 and b.shape[-1] == 109:
                # Encode first
                a_t = torch.from_numpy(a).float().unsqueeze(0)
                b_t = torch.from_numpy(b).float().unsqueeze(0)
                emb_a = self.forward(a_t)
                emb_b = self.forward(b_t)
            else:
                emb_a = torch.from_numpy(a).float().unsqueeze(0)
                emb_b = torch.from_numpy(b).float().unsqueeze(0)
            sim = F.cosine_similarity(emb_a, emb_b)
            return float(sim.item())
