# =========================================================================
# Copyright (C) 2026. Cloud-Device Recommendation System.
# =========================================================================

"""
Compact Embedding with Index Remapping

Provides RemappedEmbedding — a drop-in replacement for nn.Embedding that uses a
pre-computed lookup table to remap sparse, high-cardinality indices into a compact
contiguous range, dramatically reducing embedding table size.

Usage:
    After model construction, call apply_vocab_pruning_to_model() to replace
    oversized nn.Embedding layers with RemappedEmbedding instances.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from cloud_device_recsys.data.vocab_pruner import VocabPruneInfo

logger = logging.getLogger(__name__)


class RemappedEmbedding(nn.Module):
    """
    A compact embedding layer that remaps input indices before lookup.

    Stores a small nn.Embedding(compact_vocab_size, emb_dim) and a buffer
    remap_table that maps original (sparse) indices to compact (dense) indices.

    Indices not seen during training are mapped to padding_idx (0).
    """

    def __init__(
        self,
        compact_embedding: nn.Embedding,
        remap_table: torch.LongTensor,
        original_vocab_size: int,
    ):
        super().__init__()
        self.embedding = compact_embedding
        
        # Register as buffer: moves with model to GPU, saved in state_dict,
        # but not a parameter (no gradients)
        self.register_buffer("remap_table", remap_table)
        self.original_vocab_size = original_vocab_size
        self.compact_vocab_size = compact_embedding.num_embeddings
        self.embedding_dim = compact_embedding.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp to valid range to handle any OOV indices gracefully
        clamped = x.clamp(0, self.original_vocab_size - 1)
        remapped = self.remap_table[clamped]
        return self.embedding(remapped)

    @property
    def weight(self):
        """Expose weight for compatibility with code that accesses embedding.weight."""
        return self.embedding.weight

    def extra_repr(self) -> str:
        return (
            f"original_vocab={self.original_vocab_size}, "
            f"compact_vocab={self.compact_vocab_size}, "
            f"dim={self.embedding_dim}, "
            f"compression={1 - self.compact_vocab_size / self.original_vocab_size:.1%}"
        )


def apply_vocab_pruning_to_model(model, prune_info: VocabPruneInfo, feature_map=None):
    """
    Replace nn.Embedding layers in the model with RemappedEmbedding for prunable features.

    Walks the model's embedding layers (typically in FeatureEmbeddingDict.embedding_layers)
    and replaces oversized nn.Embedding with compact RemappedEmbedding.

    Handles shared embeddings: features sharing the same embedding get the
    same RemappedEmbedding reference.

    Args:
        model: FuxiCTR model instance (e.g., DCNv3, PNN, FCN)
        prune_info: VocabPruneInfo from build_vocab_mapping()
        feature_map: Optional FeatureMap for share_embedding info
    """
    if not prune_info or not prune_info.features:
        logger.info("[VocabPruner] No features to prune in model.")
        return

    # Find all ModuleDict instances that hold embedding layers
    embedding_dicts = _find_embedding_dicts(model)
    if not embedding_dicts:
        logger.warning("[VocabPruner] No embedding ModuleDict found in model.")
        return

    total_old_params = 0
    total_new_params = 0
    replaced_count = 0

    # Track replaced base embeddings to handle shared references
    replaced_bases = {}

    for emb_dict_name, emb_dict in embedding_dicts:
        for feature_name in list(emb_dict.keys()):
            if feature_name not in prune_info.features:
                continue

            old_emb = emb_dict[feature_name]
            if not isinstance(old_emb, nn.Embedding):
                continue

            info = prune_info.features[feature_name]
            base_name = info.feature_name

            # Check if this is a shared embedding that's already been replaced
            if base_name in replaced_bases:
                emb_dict[feature_name] = replaced_bases[base_name]
                replaced_count += 1
                continue

            old_vocab = old_emb.num_embeddings
            emb_dim = old_emb.embedding_dim

            # Because feature_map was pruned before model construction,
            # FuxiCTR's FeatureEmbeddingDict already created old_emb with the compact_vocab_size!
            # It also already applied the correct initialization (e.g. N(0, 1e-4) or pretrained).
            # We simply need to wrap it and provide the remap_table.
            
            if old_vocab != info.compact_vocab_size:
                logger.warning(
                    f"Unexpected vocab size mismatch for {feature_name}: "
                    f"model has {old_vocab}, expected compact size {info.compact_vocab_size}. "
                    "Did pruning run before model construction?"
                )

            # Move remap table to the correct device
            device = old_emb.weight.device
            remap_table_device = info.remap_table.to(device)

            # Create RemappedEmbedding wrapping the ALREADY CORRECT compact embedding
            remapped_emb = RemappedEmbedding(
                compact_embedding=old_emb,
                remap_table=remap_table_device,
                original_vocab_size=info.original_vocab_size
            )

            emb_dict[feature_name] = remapped_emb
            replaced_bases[base_name] = remapped_emb
            replaced_count += 1

            old_params = info.original_vocab_size * emb_dim
            new_params = info.compact_vocab_size * emb_dim
            total_old_params += old_params
            total_new_params += new_params

            logger.info(
                f"[VocabPruner] Replaced {feature_name}: "
                f"Embedding({info.original_vocab_size}, {emb_dim}) → "
                f"RemappedEmbedding({info.compact_vocab_size}, {emb_dim}) "
                f"[{old_params:,} → {new_params:,} params]"
            )

    if replaced_count > 0:
        saved = total_old_params - total_new_params
        logger.info(
            f"[VocabPruner] Total: replaced {replaced_count} embeddings, "
            f"saved {saved:,} parameters "
            f"(~{saved * 4 / 1e6:.1f}MB in float32)"
        )
    else:
        logger.info("[VocabPruner] No embeddings were replaced.")


def _find_embedding_dicts(model):
    """
    Find all nn.ModuleDict instances that could contain embedding layers.

    Searches recursively through the model for ModuleDict instances
    that contain nn.Embedding layers (typical pattern in FuxiCTR models:
    FeatureEmbeddingDict.embedding_layers).

    Returns:
        List of (name, ModuleDict) tuples
    """
    results = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleDict):
            # Check if it contains at least one Embedding
            has_embedding = any(
                isinstance(child, nn.Embedding) for child in module.values()
            )
            if has_embedding:
                results.append((name, module))
    return results
