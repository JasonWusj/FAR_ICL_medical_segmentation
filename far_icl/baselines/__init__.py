"""Frozen-backbone selectors on L2-normalized ResNet embeddings."""

import torch
import torch.nn.functional as F


def mmr_indices(query_vector, candidate_vectors, k, relevance_weight):
    """Start with the nearest case, then penalize maximum similarity to any selected case."""
    relevance = candidate_vectors @ query_vector
    chosen = [int(relevance.argmax())]
    redundancy = torch.full_like(relevance, -torch.inf)
    for _ in range(1, k):
        redundancy = torch.maximum(redundancy, candidate_vectors @ candidate_vectors[chosen[-1]])
        scores = relevance_weight * relevance - (1 - relevance_weight) * redundancy
        scores[chosen] = -torch.inf
        chosen.append(int(scores.argmax()))
    return chosen


def dual_similarity_indices(query_vector, candidate_vectors, query_shape, candidate_shapes, k):
    """Adapted deterministic sum of semantic cosine and predicted/annotated shape cosine."""
    scores = candidate_vectors @ query_vector + F.cosine_similarity(query_shape[None], candidate_shapes)
    return scores.argsort(descending=True, stable=True)[:k].tolist()
