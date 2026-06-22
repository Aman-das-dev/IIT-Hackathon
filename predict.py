import os
import argparse
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import JigsawDataset
from model import JigsawSolver

def parse_args():
    parser = argparse.ArgumentParser(description="Predict jigsaw puzzle permutations")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to test images folder (e.g. data/puzzle_3x3/test)")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to test.csv")
    parser.add_argument('--model_path', type=str, default='model.pth', help="Path to trained model")
    parser.add_argument('--output', type=str, default='predictions.csv', help="Output CSV file path")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for inference")
    return parser.parse_args()

def predict():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = JigsawSolver()
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: Model file {args.model_path} not found. Using untrained model.")
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print("Loading test dataset...")
    # For prediction, we use is_train=False so it yields puzzle_id instead of labels
    test_dataset = JigsawDataset(args.data_dir, args.csv_file, transform=transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    results = []
    
    with torch.no_grad():
        for tiles, puzzle_ids in test_loader:
            tiles = tiles.to(device)
            outputs = model(tiles) # (batch_size, 9, 9)
            predictions = torch.argmax(outputs, dim=2) # (batch_size, 9)
            
            preds_list = predictions.cpu().tolist()
            
            for pid, pred in zip(puzzle_ids, preds_list):
                results.append([pid] + pred)
        
    # Save to CSV
    columns = ['puzzle_id', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7', 'p_8']
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    predict()
