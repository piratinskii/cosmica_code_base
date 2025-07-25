# COSMICA Dataset - Code Repository

This repository contains all the code used to create and evaluate the COSMICA astronomical dataset as described in the paper "Curated astronomical imagery for multi-class object detection evaluation".

## Data Collection and Preprocessing

**Important Note:** Image preprocessing (resize to 640×640, grayscale conversion) and data augmentation (flipping, rotation, brightness/exposure adjustment, blur, noise) were performed using the Roboflow platform, as described in the papers. The training scripts in this repository contain only basic transformations (resize + normalization) for model input.

### `forum_image_downloader.py`
Automated web scraper for collecting astronomical images from the astronomy.ru forum threads. Features:
- Downloads images from specified page ranges
- Handles forum pagination automatically  
- Includes rate limiting to avoid overwhelming the server
- Saves images with metadata tracking

**Usage:**
```bash
python forum_image_downloader.py --url "https://astronomy.ru/forum/index.php/topic,16429" --start 1 --end 10 --delay 1.0
```

### `background_generator.py`
Generates background (empty sky) image patches from annotated astronomical images. Implements the sliding-grid algorithm described in the paper:
- Parses YOLO annotation files to identify object locations
- Uses grid search to find largest empty 640×640 pixel regions
- Extracts background patches that don't overlap with any celestial objects
- Essential for training robust models that minimize false positives

**Usage:**
```bash
python background_generator.py --images /path/to/images --labels /path/to/labels --out /path/to/output
```

## Dataset Management

### `balancer.py`
Creates balanced train/validation/test splits from the collected images. Features:
- Maintains proportional class distribution across splits
- Simple, efficient greedy balancing algorithm 
- Supports custom split ratios (default: 79%/11%/11%)
- Multiple image format support (.jpg, .png, etc.)
- Detailed statistics and class distribution reporting

**Usage:**
```bash
python balancer.py
# Enter dataset path and desired split percentages when prompted
```

### `check_balance.py`
Validates the class distribution across dataset splits. Provides:
- Detailed statistics for each object class per split
- Background image counting
- Class imbalance analysis
- Summary tables and visualizations

**Usage:**
```bash
python check_balance.py
# Enter dataset path when prompted
```

## Model Training

### `yolo_training.py`
Complete training script for YOLO family models (YOLOv8, YOLOv9, YOLOv11). Features:
- Support for all YOLO model variants (nano to extra-large)
- Automatic dataset configuration generation
- Hyperparameters matching paper methodology
- Data augmentation pipeline (rotation ±45°, flips, etc.)
- Mixed precision training for efficiency
- Checkpoint saving and progress monitoring

**Usage:**
```bash
python yolo_training.py --model yolo11n --data /path/to/cosmica --epochs 100
```

### `yolo_bootstrap_evaluation.py`
Advanced bootstrap statistical evaluation for YOLO models:
- Bootstrap confidence intervals for all metrics
- Per-class performance analysis
- Statistical significance testing
- Comprehensive visualization suite
- Model comparison framework

### `efficientdet_training.py`
Training script for EfficientDet-Lite0 model. Includes:
- COCO format dataset loading
- Custom augmentation pipeline
- Early stopping and learning rate scheduling
- Optional Comet ML experiment tracking
- Mixed precision training support

**Usage:**
```bash
python efficientdet_training.py --data /path/to/coco/dataset --epochs 100 --comet-key YOUR_API_KEY
```

### `fasterrcnn_training.py`
Training script for MobileNetV3-FasterRCNN-FPN model. Features:
- MobileNetV3 backbone for efficiency
- Anchor-based detection with FPN
- COCO evaluation metrics
- GPU acceleration support

**Usage:**
```bash
python fasterrcnn_training.py --data /path/to/coco/dataset --epochs 100 --comet-key YOUR_API_KEY
```

## Model Evaluation

### `yolo_evaluation.py`
Advanced evaluation script for YOLO models with bootstrap statistical analysis:
- Bootstrap confidence intervals for all metrics
- Per-class performance analysis
- Statistical significance testing
- Comprehensive visualization suite
- Model comparison framework

**Metrics Calculated:**
- mAP@50, mAP@50-95
- Precision, Recall, F1-score
- Per-class average precision
- Bootstrap confidence intervals

### `efficientdet_evaluation.py`
Evaluation script for EfficientDet-Lite0 model:
- COCO-style evaluation metrics (mAP@50, mAP@50-95)
- Comprehensive performance analysis
- JSON output support

**Usage:**
```bash
python efficientdet_evaluation.py --model best_model.pth --data /path/to/dataset --output metrics.json
```

### `fasterrcnn_evaluation.py`
Evaluation script for FasterRCNN model:
- Standard object detection metrics
- COCO evaluation protocol
- Performance benchmarking

**Usage:**
```bash
python fasterrcnn_evaluation.py --model best_model.pth --data /path/to/dataset --output metrics.json
```

## Real-World Testing

### `real_detector.py`
Script for testing trained models on real astronomical images. Features:
- Image preprocessing pipeline (grayscale conversion, resizing, padding)
- Batch inference support
- Visualization of detections with bounding boxes
- Confidence threshold and NMS parameter tuning
- Output saving for further analysis

**Usage:**
```bash
python real_detector.py --model /path/to/trained_model.pt --images /path/to/test_images --conf 0.2 --iou 0.8
```

## Requirements

The code requires the following main dependencies:
- Python 3.8+
- PyTorch 1.12+
- Ultralytics YOLO
- OpenCV
- Pillow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Albumentations
- pycocotools
- BeautifulSoup4 (for web scraping)
- requests
- tqdm
- scipy

## File Organization

```
REPOSITORY/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── background_generator.py        # Background image generation
├── balancer.py                    # Dataset splitting
├── check_balance.py              # Class distribution analysis
├── forum_image_downloader.py     # Automated image collection
├── yolo_training.py              # YOLO models training
├── yolo_bootstrap_evaluation.py  # YOLO bootstrap evaluation
├── efficientdet_training.py      # EfficientDet training
├── fasterrcnn_training.py        # FasterRCNN training
├── efficientdet_evaluation.py    # EfficientDet evaluation
├── fasterrcnn_evaluation.py      # FasterRCNN evaluation
└── real_detector.py              # Real-world testing
```

## License

This code is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. 
