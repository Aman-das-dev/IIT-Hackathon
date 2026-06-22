# AI Jigsaw Solver Approach Report

## Approach
This project leverages deep learning for solving the 3x3 jigsaw puzzle challenge. We treat the reconstruction of a 3x3 jigsaw puzzle as a classification task for each individual tile.

1. **Feature Extraction**: We use a pre-trained Vision Transformer (ViT-Base-16) model as the backbone to extract high-level feature representations from each of the 9 image tiles independently. ViT is naturally suited for extracting structural details from image patches.
2. **Context Aggregation**: The extracted 768-dimensional feature vectors from all 9 tiles are concatenated to form a robust global representation of the entire puzzle set.
3. **Prediction Head**: A multi-layer perceptron (MLP) with GELU activations and Dropout acts as the classification head. It takes the combined feature vector and outputs logits for each of the 9 tiles across 9 possible grid positions.
4. **Output Generation**: During inference, the argmax of the logits for each tile is taken to assign it a position from 0 to 8.

## Novelty
- **Vision Transformer Backbone**: Shifting from traditional CNNs to a Vision Transformer (ViT) allows the model to capture fine-grained textures and long-range dependencies better, which is crucial for identifying matching puzzle edges.
- **Global Context Awareness**: Aggregating features globally before classification allows the model to understand the entire puzzle context simultaneously, rather than relying on pairwise heuristic comparisons.
- **End-to-End Predictability**: This architecture forms a fully differentiable, end-to-end baseline for layout prediction without complex post-processing rules.
