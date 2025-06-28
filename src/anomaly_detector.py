import torch
from typing import List, Tuple, Any


def detect_anomalies(embeddings: torch.Tensor, node_ids: List[Any], top_k: int = 10) -> List[Tuple[Any, float]]:
    """Identify anomalies based on distance from the mean embedding.

    Parameters
    ----------
    embeddings : torch.Tensor
        Node embeddings tensor of shape ``[num_nodes, emb_dim]``.
    node_ids : List[Any]
        List mapping each embedding to the corresponding node identifier.
    top_k : int, optional
        Number of top anomalies to return, by default 10.

    Returns
    -------
    List[Tuple[Any, float]]
        List of tuples containing node id and anomaly score sorted by
        descending score.
    """
    if embeddings.numel() == 0:
        return []

    mean_vec = embeddings.mean(dim=0)
    distances = torch.norm(embeddings - mean_vec, dim=1)
    scores = distances.cpu().tolist()

    scored_nodes = list(zip(node_ids, scores))
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    return scored_nodes[:top_k]
