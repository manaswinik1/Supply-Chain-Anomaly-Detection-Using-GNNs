# Supply Chain Anomaly Detection Using GNNs

This project demonstrates how Graph Neural Networks (GNNs) can be leveraged to detect anomalies and risky nodes in supply chain networks. The code builds a graph from simulated CSV data, trains a GNN model to learn node embeddings, scores anomalies, and visualizes results in an interactive Streamlit dashboard.

## Features
- **Graph-based modeling** of supply chain nodes and edges
- **Node embeddings** generated with a Graph Convolutional Network
- **Anomaly detection** using statistical distance from mean embeddings
- **Interactive Streamlit dashboard** for visual exploration

## Dataset
The project expects pre-simulated CSV files located in `data/raw/`:

- `supply_chain_nodes.csv` – node information (`node_id`, `type`, `location`, `risk_score`)
- `supply_chain_edges.csv` – edge relationships (`source`, `target`, `weight`, `delay`)

These files are placeholders for synthetic data and do not contain real supply chain intelligence.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit dashboard:
   ```bash
   streamlit run app/streamlit_dashboard.py
   ```

## Folder Structure
```
├── data/
│   └── raw/
│       ├── supply_chain_nodes.csv
│       └── supply_chain_edges.csv
├── models/
├── src/
│   ├── graph_loader.py
│   ├── gnn_model.py
│   ├── anomaly_detector.py
│   └── visualizer.py
├── app/
│   └── streamlit_dashboard.py
├── requirements.txt
└── README.md
```

## Screenshot
![dashboard screenshot](docs/screenshot.png)

## License
This project is provided for demonstration purposes only using simulated data.
