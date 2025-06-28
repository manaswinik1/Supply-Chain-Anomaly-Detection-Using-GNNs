import pandas as pd
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from typing import Tuple


def load_graph(node_path: str, edge_path: str) -> Tuple[nx.DiGraph, 'Data']:
    """Load node and edge CSV files and build NetworkX and PyG graphs.

    Parameters
    ----------
    node_path : str
        Path to the nodes CSV file.
    edge_path : str
        Path to the edges CSV file.

    Returns
    -------
    Tuple[nx.DiGraph, Data]
        A tuple of the NetworkX directed graph and the corresponding
        PyTorch Geometric Data object.
    """
    # Read data
    nodes_df = pd.read_csv(node_path)
    edges_df = pd.read_csv(edge_path)

    # Build networkx directed graph
    G = nx.DiGraph()

    # Encode node type as categorical integer for features/labels
    type_categories = {t: i for i, t in enumerate(nodes_df['type'].unique())}

    for _, row in nodes_df.iterrows():
        node_id = row['node_id']
        node_type = row['type']
        attrs = {
            'node_id': node_id,
            'type': node_type,
            'location': row.get('location', ''),
            'risk_score': float(row.get('risk_score', 0.0)),
            # Features for PyG
            'x': torch.tensor([
                float(row.get('risk_score', 0.0)),
                float(type_categories[node_type])
            ], dtype=torch.float),
            # Label for node classification (type)
            'y': torch.tensor(type_categories[node_type], dtype=torch.long),
        }
        G.add_node(node_id, **attrs)

    for _, row in edges_df.iterrows():
        G.add_edge(
            row['source'],
            row['target'],
            weight=float(row.get('weight', 1.0)),
            delay=float(row.get('delay', 0.0)),
        )

    # Convert to PyTorch Geometric Data object
    data = from_networkx(G)

    # Ensure correct tensor types
    data.x = torch.stack(list(data.x))
    data.y = torch.tensor(data.y, dtype=torch.long)

    return G, data
