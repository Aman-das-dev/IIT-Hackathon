# AI Jigsaw Solver Approach Report

## Overview
**AI JIGSAW SOLVER: Intelligent Puzzle Reconstruction System**
*(Python, PyTorch, CNN, GNN, Computer Vision)*

### The Challenge
Built an AI-based system to reconstruct shuffled 3×3 image puzzles by identifying visual and spatial relationships between image tiles.

### Engineering Approach
1. **Feature Extraction (CNN)**: Used Convolutional Neural Networks (ResNet-18 backbone) to extract high-level visual features such as texture, color, and edges independently from each of the 9 image tiles.
2. **Context Aggregation (GNN)**: Applied Graph Neural Networks (GNN) to model tile adjacency and spatial dependencies. By treating each tile as a node in a fully connected graph, the GNN learns the complex pairwise relationships and message passing between potential adjacent edges.
3. **Hybrid Architecture**: Combined the CNN feature extractor with the GNN relational model for a powerful hybrid puzzle reconstruction network.
4. **Optimization**: Optimized performance using the Adam optimizer, Hyperparameter tuning, and extensive experiments to enhance model performance.

### Key Outcomes
- Achieved approximately **67% reconstruction accuracy**, improving performance from an initial 14% baseline to over 64%.
- Secured **3rd Prize** at the IIT Bhubaneswar ML Hackathon.

## Architecture Details
- **CNN Backbone**: A pre-trained ResNet-18 (with the final classification layers removed) outputs a dense feature vector for each of the 9 tiles.
- **GNN Layers**: A Graph Attention mechanism processes the 9 node vectors, allowing tiles to exchange structural information. 
- **Classification Head**: A Multi-Layer Perceptron (MLP) outputs the final logits, predicting the true absolute position (0-8) for each shuffled tile in the original grid.
