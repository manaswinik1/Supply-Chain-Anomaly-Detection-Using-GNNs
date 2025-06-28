import networkx as nx
import matplotlib.pyplot as plt
from typing import Iterable, Any


def plot_graph(graph: nx.DiGraph, anomalies: Iterable[Any] = None):
    """Plot the supply chain graph highlighting anomaly nodes.

    Parameters
    ----------
    graph : nx.DiGraph
        NetworkX graph representing the supply chain.
    anomalies : Iterable[Any], optional
        Collection of node ids considered anomalous, by default None.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object for further use (e.g. in Streamlit).
    """
    if anomalies is None:
        anomalies = []

    pos = nx.spring_layout(graph)
    node_colors = ["red" if n in anomalies else "skyblue" for n in graph.nodes()]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    ax.set_axis_off()
    fig.tight_layout()
    return fig
