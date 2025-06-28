import streamlit as st
from pathlib import Path

from src.graph_loader import load_graph
from src.gnn_model import train_gnn
from src.anomaly_detector import detect_anomalies
from src.visualizer import plot_graph


st.set_page_config(page_title="Supply Chain GNN Anomaly Detection", layout="wide")

def load_default_data():
    node_path = Path("data/raw/supply_chain_nodes.csv")
    edge_path = Path("data/raw/supply_chain_edges.csv")
    return load_graph(node_path, edge_path)


def main():
    st.title("Supply Chain Anomaly Detection Using GNNs")

    with st.sidebar:
        st.header("Data Files")
        node_file = st.text_input("Nodes CSV", "data/raw/supply_chain_nodes.csv")
        edge_file = st.text_input("Edges CSV", "data/raw/supply_chain_edges.csv")
        run_btn = st.button("Run Detection")

    if run_btn:
        graph, data = load_graph(node_file, edge_file)
        model, embeddings = train_gnn(data)
        anomalies = detect_anomalies(embeddings, list(graph.nodes()))

        st.subheader("Anomaly Scores")
        st.table(anomalies)

        st.subheader("Supply Chain Graph")
        fig = plot_graph(graph, [a[0] for a in anomalies])
        st.pyplot(fig)
    else:
        st.info("Provide data file paths and click 'Run Detection' to begin.")


if __name__ == "__main__":
    main()
