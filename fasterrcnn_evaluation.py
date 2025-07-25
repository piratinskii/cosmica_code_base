#!/usr/bin/env python3
"""
MobileNetV3-FasterRCNN-FPN Evaluation Script for COSMICA Dataset

Evaluates trained FasterRCNN model on test dataset.
Calculates COCO-style metrics including mAP@50 and mAP@50-95.

Usage: python fasterrcnn_evaluation.py --model path/to/model.pth --data path/to/dataset
"""

import torch
import argparse
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

# Import from training script (assumes it's in the same directory)
try:
    from fasterrcnn_training import COCODataset, get_transforms, collate_fn, get_fasterrcnn_model
except ImportError:
    print("Error: Could not import from fasterrcnn_training.py")
    print("Make sure fasterrcnn_training.py is in the same directory")
    exit(1)


def evaluate_fasterrcnn(model_path, data_path, device='auto'):
    """Evaluate FasterRCNN model on test dataset"""
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Data paths
    test_dir = os.path.join(data_path, 'test')
    test_anno = os.path.join(test_dir, '_annotations.coco.json')
    
    # Create test dataset
    test_dataset = COCODataset(test_dir, test_anno, transforms=get_transforms(False))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    
    # Load model
    num_classes = test_dataset.get_num_classes()
    model = get_fasterrcnn_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded with {num_classes} classes")
    print(f"Evaluating on {len(test_dataset)} test images")
    
    # Run evaluation
    predictions = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                image_id = target["image_id"].item()
                
                if len(output['boxes']) == 0:
                    continue
                
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        "score": float(score)
                    })
    
    # Evaluate using COCO API
    if not predictions:
        print("No predictions found!")
        return 0.0
    
    with open('temp_predictions.json', 'w') as f:
        json.dump(predictions, f)
    
    try:
        coco_gt = COCO(test_anno)
        coco_dt = coco_gt.loadRes('temp_predictions.json')
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract key metrics
        metrics = {
            'mAP@50-95': coco_eval.stats[0],
            'mAP@50': coco_eval.stats[1],
            'mAP@75': coco_eval.stats[2],
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5]
        }
        
        print("\nEvaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None
    finally:
        if os.path.exists('temp_predictions.json'):
            os.remove('temp_predictions.json')


def main():
    parser = argparse.ArgumentParser(description='Evaluate MobileNetV3-FasterRCNN-FPN on COSMICA test set')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--data', required=True, help='Path to COCO dataset directory')
    parser.add_argument('--device', default='auto', help='Device to use for evaluation')
    parser.add_argument('--output', help='Output file for metrics (JSON format)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        exit(1)
    
    if not os.path.exists(args.data):
        print(f"Error: Dataset directory not found: {args.data}")
        exit(1)
    
    # Run evaluation
    metrics = evaluate_fasterrcnn(args.model, args.data, args.device)
    
    if metrics and args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main() 