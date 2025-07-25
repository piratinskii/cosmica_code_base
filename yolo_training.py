#!/usr/bin/env python3
"""
YOLO Training Script for COSMICA Dataset

Trains YOLOv8, YOLOv9, or YOLOv11 models on astronomical object detection.
Supports all model variants described in the paper.
"""

import argparse
import os
import sys
from ultralytics import YOLO
import torch


def train_yolo_model(model_name, dataset_path, epochs=100, batch_size=16, imgsz=640, device='auto'):
    """
    Train a YOLO model on the COSMICA dataset
    
    Args:
        model_name: YOLO model variant ('yolov8n', 'yolov9c', 'yolo11n', etc.)
        dataset_path: Path to dataset directory containing dataset.yaml
        epochs: Number of training epochs (default: 100)
        batch_size: Training batch size (default: 16)
        imgsz: Input image size (default: 640)
        device: Training device ('auto', 'cpu', 'cuda', 0, 1, etc.)
    """
    
    print(f"Starting training: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}")
    print(f"Device: {device}")
    
    # Load pretrained model
    model = YOLO(f"{model_name}.pt")
    
    # Find dataset.yaml file
    dataset_yaml = os.path.join(dataset_path, "dataset.yaml")
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"dataset.yaml not found in {dataset_path}")
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project='runs/train',
        name=f'{model_name}_cosmica',
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        cache=True,  # Cache images for faster training
        amp=True,  # Automatic Mixed Precision
        # Hyperparameters matching paper methodology
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,  # Box loss gain
        cls=0.5,  # Classification loss gain
        dfl=1.5   # Distribution focal loss gain
    )
    
    print(f"Training completed! Results saved in: runs/train/{model_name}_cosmica")
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO models on COSMICA dataset')
    parser.add_argument('--model', required=True, 
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                               'yolov9t', 'yolov9s', 'yolov9m', 'yolov9c', 'yolov9e',
                               'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
                       help='YOLO model variant to train')
    parser.add_argument('--data', required=True, help='Path to COSMICA dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--device', default='auto', help='Training device (auto, cpu, cuda, 0, 1, etc.)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.data):
        print(f"Error: Dataset path not found: {args.data}")
        sys.exit(1)
    
    # Check for dataset.yaml
    dataset_yaml = os.path.join(args.data, "dataset.yaml")
    if not os.path.exists(dataset_yaml):
        print(f"Error: dataset.yaml not found in {args.data}")
        sys.exit(1)
    
    # Check for required subdirectories
    required_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
    for dir_path in required_dirs:
        full_path = os.path.join(args.data, dir_path)
        if not os.path.exists(full_path):
            print(f"Error: Required directory not found: {full_path}")
            sys.exit(1)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available. GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available. Training will use CPU (slower).")
    
    # Train the model
    try:
        train_yolo_model(
            model_name=args.model,
            dataset_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device
        )
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 