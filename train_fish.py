from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import os

def train_yolo():
    # Get absolute paths
    current_dir = Path(__file__).parent.absolute()
    data_yaml = current_dir / 'data' / 'data.yaml'
    
    # Configure data
    data = {
        'path': str(current_dir / 'data'),
        'train': str(current_dir / 'data' / 'train' / 'images'),
        'val': str(current_dir / 'data' / 'valid' / 'images'),
        'test': '',  # no test set
        'nc': 1,
        'names': ['fish']
    }
    
    # Save configuration
    with open(data_yaml, 'w') as f:
        yaml.dump(data, f)

    # Create and configure model
    model = YOLO('yolov8n.pt')  # load a pretrained model

    # Training arguments
    args = {
        'data': str(data_yaml),
        'epochs': 5,
        'imgsz': 640,
        'batch': 8,
        'device': 'cpu',  # explicitly set to use CPU
        'workers': 4,  # reduced number of workers for CPU
        'name': 'fish_detector',
        'patience': 50,
        'save_period': 10,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'SGD',
        'cos_lr': True,
        'single_cls': True  # single class detection
    }

    # Start training
    try:
        print(f"Starting training with data config: {data}")
        print(f"Training on CPU - this may take longer than GPU training")
        model.train(**args)
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise e

if __name__ == '__main__':
    train_yolo() 