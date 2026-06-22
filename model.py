import torch
import torch.nn as nn
import torchvision.models as models

class GraphAttentionLayer(nn.Module):
    """
    A Graph Attention Network (GAT) layer implemented via Self-Attention.
    Treats each tile as a node in a fully connected graph.
    """
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch_size, num_nodes, feature_dim)
        attn_out, _ = self.mha(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x

class JigsawSolver(nn.Module):
    def __init__(self, num_classes=9, num_tiles=9):
        super(JigsawSolver, self).__init__()
        
        self.num_classes = num_classes
        self.num_tiles = num_tiles
        
        # CNN: Feature Extraction (ResNet-18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1]) # Remove final FC
        feature_dim = resnet.fc.in_features # 512
        
        # GNN: Spatial Dependency Modeling (Graph Attention Network)
        self.gnn1 = GraphAttentionLayer(feature_dim)
        self.gnn2 = GraphAttentionLayer(feature_dim)
        
        # Classification Head (MLP)
        self.fc = nn.Sequential(
            nn.Linear(num_tiles * feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_tiles * num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_tiles, C, H, W)
        batch_size, num_tiles, C, H, W = x.size()
        
        # Flatten batch and tiles for CNN extraction
        x = x.view(-1, C, H, W) # (batch_size * num_tiles, C, H, W)
        
        # CNN Feature Extraction
        features = self.cnn(x) # (batch_size * num_tiles, 512, 1, 1)
        features = features.view(batch_size, num_tiles, -1) # (batch_size, num_tiles, 512)
        
        # GNN Message Passing (Tile Adjacency & Spatial Dependencies)
        features = self.gnn1(features)
        features = self.gnn2(features)
        
        # Flatten for MLP
        features = features.reshape(batch_size, -1) # (batch_size, num_tiles * 512)
        
        # Predict logits
        out = self.fc(features) # (batch_size, num_tiles * num_classes)
        out = out.reshape(batch_size, num_tiles, -1) # (batch_size, num_tiles, num_classes)
        
        return out
