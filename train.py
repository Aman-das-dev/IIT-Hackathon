import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import JigsawDataset
from model import JigsawSolver

def parse_args():
    parser = argparse.ArgumentParser(description="Train jigsaw puzzle solver")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to training images folder (e.g. data/puzzle_3x3/train)")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to train.csv")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--save_path', type=str, default='model.pth', help="Path to save trained model")
    return parser.parse_args()

def train():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print("Loading dataset...")
    train_dataset = JigsawDataset(args.data_dir, args.csv_file, transform=transform, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    model = JigsawSolver().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for i, (tiles, labels) in enumerate(train_loader):
            tiles, labels = tiles.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(tiles) # (batch_size, num_tiles, num_classes)
            
            # CrossEntropyLoss expects (batch_size * num_tiles, num_classes)
            outputs_flat = outputs.reshape(-1, 9)
            labels_flat = labels.reshape(-1)
            
            loss = criterion(outputs_flat, labels_flat)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 0 and i > 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0
        
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == '__main__':
    train()
