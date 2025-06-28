import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    """Simple two-layer Graph Convolutional Network."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def train_gnn(data: Data, epochs: int = 200, lr: float = 0.01):
    """Train a GCN model on the given PyG data object.

    Parameters
    ----------
    data : Data
        PyTorch Geometric data object with ``x``, ``edge_index`` and ``y``.
    epochs : int, optional
        Number of training epochs, by default 200.
    lr : float, optional
        Learning rate, by default 0.01.

    Returns
    -------
    Tuple[GCN, torch.Tensor]
        The trained model and the node embeddings as a tensor.
    """

    model = GCN(data.num_node_features, 16, int(data.y.max().item()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    return model, embeddings
