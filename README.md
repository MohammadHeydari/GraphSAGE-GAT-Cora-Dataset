# GraphSAGE-GAT-Cora-Dataset
Graph Neural Networks (GNNs) for Node Classification using Graph Attention Networks (GAT) and GraphSAGE, implemented in PyTorch and TensorFlow

This repository contains implementations of Graph Attention Networks (GAT) and GraphSAGE using both PyTorch Geometric and TensorFlow GNN frameworks. These models are used for node classification tasks on graph-structured data.

## Files
- PyTorch_GAT_GraphSAGE.ipynb → Implementation of GAT and GraphSAGE using PyTorch Geometric (PyG)

- TensorFlow_GAT_GraphSAGE.ipynb → Implementation of GAT and GraphSAGE using TensorFlow GNN.

## Getting Started
Install Dependencies

Before running the notebooks, install the required libraries:

For PyTorch Geometric (PyG):

```pip install torch torchvision torchaudio torch-geometric```

```pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.6+cpu.html```

### For TensorFlow GNN:

```pip install tensorflow tensorflow_gnn```

## Run the Notebooks

Execute the Jupyter notebooks:

jupyter notebook

Open either PyTorch_GAT_GraphSAGE.ipynb or TensorFlow_GAT_GraphSAGE.ipynb and run the cells.

## Model Overview

### Graph Attention Networks (GAT)

GAT uses attention mechanisms to weigh the importance of neighboring nodes dynamically. This allows the model to learn more relevant node embeddings without uniform aggregation.

### GraphSAGE

GraphSAGE generates embeddings by sampling a fixed number of neighbors rather than processing the entire graph, making it scalable to large graphs.

### Dataset

Both notebooks use the Cora dataset, a widely used benchmark for node classification:

2708 nodes (papers)

5429 edges (citation relationships)

7 classes (research topics)

## References

Velickovic et al., "Graph Attention Networks", ICLR 2018 (Paper)

Hamilton et al., "Inductive Representation Learning on Large Graphs" (Paper)
