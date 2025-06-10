from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

def train_yolo():
    current_dir = Path(__file__).parent.absolute()
    data_yaml = current_dir / 'data' / 'data.yaml'
    
    data = {
        'path': str(current_dir / 'data'),
        'train': str(current_dir / 'data' / 'train' / 'images'),
        'val': str(current_dir / 'data' / 'valid' / 'images'),
        'test': '',
        'nc': 1,
        'names': ['fish']
    }
    
    with open(data_yaml, 'w') as f:
        yaml.dump(data, f)

    model = YOLO('yolov8n.pt')

    args = {
        'data': str(data_yaml),
        'epochs': 50,
        'imgsz': 640,
        'batch': 8,
        'device': 'cuda',
        'workers': 4,
        'name': 'fish_detector',
        'patience': 50,
        'save_period': 10,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'SGD',
        'cos_lr': True,
        'single_cls': True
    }
    if torch.cuda.is_available():
        model.train(**args)
    else:
        print("bez cuda nie robie i nara")

if __name__ == '__main__':
    train_yolo() 