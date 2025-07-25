#!/usr/bin/env python3
"""
EfficientDet-Lite0 Training Script for COSMICA Dataset

Trains EfficientDet-Lite0 model for astronomical object detection.
Uses COCO format annotations and includes comprehensive evaluation.

Usage: python efficientdet_training.py --data /path/to/coco/dataset --epochs 100
"""

import os
import argparse
import torch
import numpy as np
import json
from PIL import Image
import time
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import math
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# EfficientDet specific imports
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet

# Optional experiment tracking
try:
    from comet_ml import Experiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("CometML not available. Training without experiment tracking.")


class COCODataset(Dataset):
    """Custom COCO Dataset for EfficientDet training"""
    
    def __init__(self, root_dir, annotation_file, transforms=None, img_size=512):
        self.root_dir = root_dir
        self.transforms = transforms
        self.img_size = img_size

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        # Create lookup dictionaries
        self.image_dict = {img['id']: img for img in self.coco_data['images']}
        
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.img_ids = list(self.annotations.keys())
        
        # Create category mapping
        self.cat_ids = {cat['id']: i for i, cat in enumerate(self.coco_data['categories'])}
        self.cat_names = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.image_dict[img_id]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image: {img_path}")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = img.shape[:2]

        # Process annotations
        annotations = self.annotations[img_id]
        boxes = []
        labels = []

        for ann in annotations:
            x, y, w, h = ann['bbox']
            x_min, y_min = max(0, x), max(0, y)
            x_max = min(orig_width, x + w)
            y_max = min(orig_height, y + h)

            if x_min >= x_max or y_min >= y_max or w <= 1 or h <= 1:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.cat_ids[ann['category_id']])

        # Handle empty annotations
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Apply transformations
        if self.transforms:
            try:
                transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
                img = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['labels']
                
                boxes = torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
            except Exception as e:
                print(f"Transform error for {img_path}: {e}")
                img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dictionary for EfficientDet
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "img_size": torch.tensor([img.shape[1], img.shape[2]] if isinstance(img, torch.Tensor) else [orig_height, orig_width]),
            "img_scale": torch.tensor([1.0])
        }

        return img, target

    def __len__(self):
        return len(self.img_ids)

    def get_num_classes(self):
        return len(self.coco_data['categories'])


def get_transforms(img_size=512, is_training=True):
    """Get data transformations (augmentation was performed in Roboflow)"""
    if is_training:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1.0, min_visibility=0.1))
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1.0, min_visibility=0.1))


def collate_fn(batch):
    return tuple(zip(*batch))


def get_efficientdet_model(num_classes, img_size=512):
    """Create EfficientDet-Lite0 model"""
    config = get_efficientdet_config('tf_efficientdet_lite0')
    config.image_size = (img_size, img_size)
    config.num_classes = num_classes
    config.max_det_per_image = 100

    model = EfficientDet(config, pretrained_backbone=True)
    model.class_net = HeadNet(config, num_outputs=num_classes)
    
    return DetBenchTrain(model, config)


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for i, (images, targets) in enumerate(pbar):
        images = torch.stack([img.to(device) for img in images])
        
        # Prepare targets for EfficientDet
        target_dict = {
            'bbox': [t['boxes'].to(device) for t in targets],
            'cls': [t['labels'].to(device) for t in targets],
            'img_scale': torch.cat([t['img_scale'].to(device) for t in targets]),
            'img_size': torch.stack([t['img_size'].to(device) for t in targets]),
        }
        
        if scaler:
            with autocast(device_type=device.type):
                output = model(images, target_dict)
                loss = output[0] if isinstance(output, tuple) else output
        else:
            output = model(images, target_dict)
            loss = output[0] if isinstance(output, tuple) else output
        
        optimizer.zero_grad()
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(data_loader)


def evaluate_model(model, data_loader, device, annotation_file):
    """Evaluate model using COCO metrics"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = torch.stack([img.to(device) for img in images])
            
            target_dict = {
                'img_scale': torch.cat([t['img_scale'].to(device) for t in targets]),
                'img_size': torch.stack([t['img_size'].to(device) for t in targets]),
            }
            
            outputs = model(images, target_dict)
            
            # Convert outputs to COCO format
            if isinstance(outputs, dict) and 'detections' in outputs:
                detections = outputs['detections']
            else:
                detections = outputs
            
            for detection, target in zip(detections, targets):
                if detection is None or len(detection) == 0:
                    continue
                    
                image_id = target["image_id"].item()
                boxes = detection[:, :4].cpu().numpy()
                scores = detection[:, 4].cpu().numpy()
                classes = detection[:, 5].cpu().numpy().astype(int)
                
                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(cls) + 1,
                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        "score": float(score)
                    })
    
    if not predictions:
        return 0.0
    
    # Evaluate using COCO API
    with open('temp_predictions.json', 'w') as f:
        json.dump(predictions, f)
    
    try:
        coco_gt = COCO(annotation_file)
        coco_dt = coco_gt.loadRes('temp_predictions.json')
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map_score = coco_eval.stats[0]
    except Exception as e:
        print(f"Evaluation error: {e}")
        map_score = 0.0
    finally:
        if os.path.exists('temp_predictions.json'):
            os.remove('temp_predictions.json')
    
    return map_score


def main():
    parser = argparse.ArgumentParser(description='Train EfficientDet-Lite0 on COSMICA dataset')
    parser.add_argument('--data', required=True, help='Path to COCO dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=512, help='Input image size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--device', default='auto', help='Training device')
    parser.add_argument('--comet-key', help='CometML API key (optional)')
    parser.add_argument('--comet-project', default='efficientdet-cosmica', help='CometML project name')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup experiment tracking
    experiment = None
    if COMET_AVAILABLE and args.comet_key:
        os.environ["COMET_API_KEY"] = args.comet_key
        experiment = Experiment(project_name=args.comet_project)
        experiment.log_parameters(vars(args))
    
    # Data paths
    train_dir = os.path.join(args.data, 'train')
    valid_dir = os.path.join(args.data, 'valid')
    train_anno = os.path.join(train_dir, '_annotations.coco.json')
    valid_anno = os.path.join(valid_dir, '_annotations.coco.json')
    
    # Create datasets
    train_dataset = COCODataset(train_dir, train_anno, 
                               transforms=get_transforms(args.img_size, True), 
                               img_size=args.img_size)
    valid_dataset = COCODataset(valid_dir, valid_anno,
                               transforms=get_transforms(args.img_size, False),
                               img_size=args.img_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Create model
    num_classes = train_dataset.get_num_classes()
    model = get_efficientdet_model(num_classes, args.img_size).to(device)
    
    print(f"Model created with {num_classes} classes")
    print(f"Training on {len(train_dataset)} images, validating on {len(valid_dataset)} images")
    
    # Setup training
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.937, weight_decay=0.0005)
    scaler = GradScaler() if device.type == 'cuda' else None
    
    best_map = 0.0
    patience_counter = 0
    patience = 20
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        
        # Evaluate
        map_score = evaluate_model(model, valid_loader, device, valid_anno)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={train_loss:.4f}, mAP={map_score:.4f}")
        
        # Log metrics
        if experiment:
            experiment.log_metrics({
                "train_loss": train_loss,
                "val_mAP": map_score
            }, epoch=epoch)
        
        # Save best model
        if map_score > best_map:
            best_map = map_score
            patience_counter = 0
            torch.save(model.state_dict(), f'best_efficientdet_map_{map_score:.4f}.pth')
            print(f"New best model saved: mAP={map_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training completed. Best mAP: {best_map:.4f}")
    
    if experiment:
        experiment.end()


if __name__ == "__main__":
    main() 