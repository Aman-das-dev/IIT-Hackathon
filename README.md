# AI Jigsaw Solver

This project provides a baseline solution for the AI Jigsaw Solver Hackathon.

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. To train the model, you need a training dataset structure like:
   ```
   train_data/
     puzzle_01/
       0.jpg
       1.jpg
       ...
       8.jpg
     puzzle_02/
       ...
   ```
   Run training:
   ```bash
   python train.py --train_dir path/to/train_data --epochs 10 --batch_size 16
   ```

3. To generate predictions on test data (which should follow the same structure as training data):
   ```bash
   python predict.py --test_dir path/to/test_data --model_path model.pth --output predictions.csv
   ```

## Files
- `model.py`: Contains the ResNet-based feature extractor and classification head.
- `dataset.py`: PyTorch Dataset class for loading 9 image tiles.
- `train.py`: Training script.
- `predict.py`: Prediction script that outputs a CSV file as per competition requirements.
