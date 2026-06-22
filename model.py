import torch
import torch.nn as nn
import torchvision.models as models

class JigsawSolver(nn.Module):
    def __init__(self, num_tiles=9, num_classes=9):
        super(JigsawSolver, self).__init__()
        # Use a pretrained Vision Transformer (ViT) to extract features
        # ViT is naturally suited for patch/tile-based understanding
        vit = models.vit_b_16(pretrained=True)
        
        # We'll use the ViT without its classification head
        self.feature_extractor = vit
        self.feature_extractor.heads = nn.Identity() # Remove the final classification layer
        
        feature_dim = 768 # ViT-Base hidden size
        
        # We process 9 tiles, concatenate their features, and predict 9 classes for each of the 9 tiles
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * num_tiles, 2048),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_tiles * num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, num_tiles, C, H, W)
        batch_size, num_tiles, C, H, W = x.size()
        
        # Flatten the batch and tile dimensions
        x = x.view(-1, C, H, W)
        
        # Extract features using ViT
        features = self.feature_extractor(x) # (batch_size * num_tiles, feature_dim)
        features = features.reshape(batch_size, num_tiles, -1) # (batch_size, num_tiles, feature_dim)
        
        # Flatten features of all tiles
        features = features.reshape(batch_size, -1) # (batch_size, num_tiles * feature_dim)
        
        # Predict logits
        out = self.fc(features) # (batch_size, num_tiles * num_classes)
        out = out.reshape(batch_size, num_tiles, -1) # (batch_size, num_tiles, num_classes)
        
        return out
