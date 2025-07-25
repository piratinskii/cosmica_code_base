#!/usr/bin/env python3
"""
YOLO Bootstrap Evaluation for COSMICA Dataset

Comprehensive bootstrap statistical evaluation of YOLO models with confidence intervals,
statistical significance testing, and detailed visualizations.

Usage: python yolo_bootstrap_evaluation.py
"""

import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import time
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings('ignore')

class BootstrapEvaluator:
    def __init__(self, model_paths, dataset_path, n_bootstrap=1000, confidence_level=0.95):
        """
        Initialize Bootstrap Evaluator for YOLO models
        
        Args:
            model_paths: dict with model names as keys and paths as values
            dataset_path: path to YOLO format dataset
            n_bootstrap: number of bootstrap iterations
            confidence_level: confidence level for intervals (default 0.95)
        """
        self.model_paths = model_paths
        self.dataset_path = dataset_path
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        # Class names from your paper
        self.class_names = ['Comets', 'Galaxies', 'Nebulae', 'Globular Clusters']
        
        # Results storage
        self.results = {model_name: [] for model_name in model_paths.keys()}
        
    def load_test_data(self):
        """Load test images and labels from YOLO format dataset"""
        test_images_path = Path(self.dataset_path) / 'test' / 'images'
        test_labels_path = Path(self.dataset_path) / 'test' / 'labels'
        
        image_files = sorted(list(test_images_path.glob('*.jpg')))
        label_files = []
        
        for img_file in image_files:
            label_file = test_labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                label_files.append(label_file)
            else:
                label_files.append(None)
                
        return image_files, label_files
    
    def bootstrap_sample(self, image_files, label_files):
        """Create bootstrap sample with replacement"""
        n_samples = len(image_files)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        sampled_images = [image_files[i] for i in indices]
        sampled_labels = [label_files[i] for i in indices]
        
        return sampled_images, sampled_labels
    
    def evaluate_model_on_sample(self, model, sampled_images, sampled_labels):
        """Evaluate model on bootstrap sample"""
        from collections import defaultdict
        import shutil
        
        # Create temporary directory for this bootstrap sample
        temp_dir = Path(f'temp_bootstrap_{np.random.randint(100000)}')
        temp_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (temp_dir / 'images').mkdir(exist_ok=True)
        (temp_dir / 'labels').mkdir(exist_ok=True)
        
        # Copy sampled files to temp directory
        for img_path, lbl_path in zip(sampled_images, sampled_labels):
            if img_path and lbl_path:
                shutil.copy2(img_path, temp_dir / 'images' / img_path.name)
                shutil.copy2(lbl_path, temp_dir / 'labels' / lbl_path.name)
        
        # Create yaml config with proper path handling for Windows
        yaml_content = f"""path: {str(self.dataset_path).replace(os.sep, '/')}
train: train/images
val: {str(temp_dir / 'images').replace(os.sep, '/')}
test: test/images
nc: 4
names: ['comet', 'galaxy', 'nebula', 'globular_cluster']
"""
        
        temp_yaml = f'temp_bootstrap_{np.random.randint(10000)}.yaml'
        with open(temp_yaml, 'w') as f:
            f.write(yaml_content)
        
        try:
            # Run validation on bootstrap sample
            metrics = model.val(
                data=temp_yaml,
                imgsz=640,
                batch=16,
                device='cuda',
                verbose=False,
                save=False,
                plots=False,
                exist_ok=True
            )
            
            # Extract metrics
            result = {
                'mAP50-95': float(metrics.box.map) if metrics.box.map else 0.0,
                'mAP50': float(metrics.box.map50) if metrics.box.map50 else 0.0,
                'precision': float(metrics.box.mp) if metrics.box.mp else 0.0,
                'recall': float(metrics.box.mr) if metrics.box.mr else 0.0,
            }
            
            # Add per-class metrics
            if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
                for i, class_name in enumerate(self.class_names):
                    if i < len(metrics.box.maps):
                        result[f'mAP50_{class_name}'] = float(metrics.box.maps[i])
                    else:
                        result[f'mAP50_{class_name}'] = 0.0
            else:
                for class_name in self.class_names:
                    result[f'mAP50_{class_name}'] = 0.0
                    
        finally:
            # Clean up
            if os.path.exists(temp_yaml):
                os.remove(temp_yaml)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                
        return result
    
    def measure_fps(self, model, n_samples=100):
        """Measure FPS on a subset of images"""
        image_files, _ = self.load_test_data()
        
        # Use subset for FPS measurement
        test_images = image_files[:min(n_samples, len(image_files))]
        
        # Warm up
        for _ in range(5):
            _ = model(test_images[0], verbose=False)
        
        # Measure FPS
        start_time = time.time()
        for img_path in test_images:
            _ = model(img_path, verbose=False)
        end_time = time.time()
        
        fps = len(test_images) / (end_time - start_time)
        return fps
    
    def run_bootstrap(self):
        """Run bootstrap evaluation for all models"""
        print(f"Starting bootstrap evaluation with {self.n_bootstrap} iterations...")
        
        # Load test data
        image_files, label_files = self.load_test_data()
        print(f"Loaded {len(image_files)} test images")
        
        # Evaluate each model
        for model_name, model_path in self.model_paths.items():
            print(f"\nEvaluating {model_name}...")
            
            # Load model
            model = YOLO(model_path)
            
            # Measure FPS once (it's hardware dependent, not sample dependent)
            fps = self.measure_fps(model)
            print(f"FPS for {model_name}: {fps:.2f}")
            
            # Bootstrap iterations
            for i in tqdm(range(self.n_bootstrap), desc=f"Bootstrap {model_name}"):
                # Create bootstrap sample
                sampled_images, sampled_labels = self.bootstrap_sample(image_files, label_files)
                
                # Evaluate on sample
                metrics = self.evaluate_model_on_sample(model, sampled_images, sampled_labels)
                metrics['fps'] = fps  # Add FPS to metrics
                
                self.results[model_name].append(metrics)
                
            # Clear GPU memory
            del model
            torch.cuda.empty_cache()
    
    def calculate_statistics(self):
        """Calculate bootstrap statistics for all models"""
        statistics = {}
        
        for model_name, bootstrap_results in self.results.items():
            model_stats = {}
            
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(bootstrap_results)
            
            # Calculate statistics for each metric
            for metric in df.columns:
                values = df[metric].values
                
                # Basic statistics
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                
                # Confidence intervals using percentile method
                lower_ci = np.percentile(values, (self.alpha/2) * 100)
                upper_ci = np.percentile(values, (1 - self.alpha/2) * 100)
                
                model_stats[metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': lower_ci,
                    'ci_upper': upper_ci,
                    'values': values  # Store for statistical tests
                }
            
            statistics[model_name] = model_stats
            
        return statistics
    
    def perform_statistical_tests(self, statistics):
        """Perform pairwise statistical tests between models"""
        model_names = list(statistics.keys())
        test_results = {}
        
        # Metrics to test
        metrics_to_test = ['mAP50-95', 'mAP50', 'precision', 'recall']
        
        for metric in metrics_to_test:
            test_results[metric] = {}
            
            # Pairwise comparisons
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    
                    values1 = statistics[model1][metric]['values']
                    values2 = statistics[model2][metric]['values']
                    
                    # Paired t-test (since bootstrap samples are paired)
                    t_stat, p_value = stats.ttest_rel(values1, values2)
                    
                    # Effect size (Cohen's d)
                    diff = values1 - values2
                    cohen_d = np.mean(diff) / np.std(diff, ddof=1)
                    
                    test_results[metric][f"{model1}_vs_{model2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohen_d': cohen_d,
                        'significant': p_value < 0.05
                    }
        
        return test_results
    
    def plot_results(self, statistics, save_path='bootstrap_results'):
        """Create visualizations of bootstrap results"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Main metrics comparison with confidence intervals
        metrics = ['mAP50-95', 'mAP50', 'precision', 'recall', 'fps']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            model_names = []
            means = []
            ci_lower = []
            ci_upper = []
            
            for model_name in self.model_paths.keys():
                model_names.append(model_name)
                means.append(statistics[model_name][metric]['mean'])
                ci_lower.append(statistics[model_name][metric]['ci_lower'])
                ci_upper.append(statistics[model_name][metric]['ci_upper'])
            
            # Calculate error bars
            yerr_lower = np.array(means) - np.array(ci_lower)
            yerr_upper = np.array(ci_upper) - np.array(means)
            
            # Create bar plot with error bars
            bars = ax.bar(model_names, means)
            ax.errorbar(model_names, means, yerr=[yerr_lower, yerr_upper], 
                       fmt='none', color='black', capsize=5)
            
            # Color bars based on best performer
            best_idx = np.argmax(means)
            for i, bar in enumerate(bars):
                if i == best_idx:
                    bar.set_color('green')
                else:
                    bar.set_color('lightblue')
            
            ax.set_title(f'{metric}')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-class mAP@50 comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        for i, model_name in enumerate(self.model_paths.keys()):
            means = []
            errors = []
            
            for class_name in self.class_names:
                metric_name = f'mAP50_{class_name}'
                means.append(statistics[model_name][metric_name]['mean'])
                ci_lower = statistics[model_name][metric_name]['ci_lower']
                ci_upper = statistics[model_name][metric_name]['ci_upper']
                error = (ci_upper - ci_lower) / 2
                errors.append(error)
            
            ax.bar(x + i*width, means, width, label=model_name, 
                   yerr=errors, capsize=5)
        
        ax.set_xlabel('Object Class')
        ax.set_ylabel('mAP@50')
        ax.set_title('Per-class mAP@50 with 95% Confidence Intervals')
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/per_class_map50.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Bootstrap distribution plots for key metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        key_metrics = ['mAP50-95', 'mAP50', 'precision', 'recall']
        
        for idx, metric in enumerate(key_metrics):
            ax = axes[idx]
            
            for model_name in self.model_paths.keys():
                values = statistics[model_name][metric]['values']
                ax.hist(values, bins=30, alpha=0.5, label=model_name, density=True)
            
            ax.set_xlabel(metric)
            ax.set_ylabel('Density')
            ax.set_title(f'Bootstrap Distribution of {metric}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/bootstrap_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, statistics, test_results, save_path='bootstrap_results'):
        """Save all results to files"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Save summary statistics
        summary = {}
        for model_name, model_stats in statistics.items():
            summary[model_name] = {}
            for metric, values in model_stats.items():
                summary[model_name][metric] = {
                    'mean': float(values['mean']),
                    'std': float(values['std']),
                    'ci_lower': float(values['ci_lower']),
                    'ci_upper': float(values['ci_upper'])
                }
        
        with open(f'{save_path}/summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        # 2. Save statistical test results
        with open(f'{save_path}/statistical_tests.json', 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # 3. Create LaTeX table for the paper
        self.create_latex_table(statistics, save_path)
        
        # 4. Save raw bootstrap results
        for model_name, results in self.results.items():
            df = pd.DataFrame(results)
            df.to_csv(f'{save_path}/bootstrap_raw_{model_name}.csv', index=False)
    
    def create_latex_table(self, statistics, save_path):
        """Create LaTeX table with results and confidence intervals"""
        
        # Main metrics table
        latex_table = "\\begin{table}[h]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{Bootstrap evaluation results with 95\\% confidence intervals (n=1000)}\n"
        latex_table += "\\begin{tabular}{lccccc}\n"
        latex_table += "\\hline\n"
        latex_table += "Model & mAP@50-95 & mAP@50 & Precision & Recall & FPS \\\\\n"
        latex_table += "\\hline\n"
        
        for model_name in self.model_paths.keys():
            row = f"{model_name}"
            
            for metric in ['mAP50-95', 'mAP50', 'precision', 'recall', 'fps']:
                mean = statistics[model_name][metric]['mean']
                ci_lower = statistics[model_name][metric]['ci_lower']
                ci_upper = statistics[model_name][metric]['ci_upper']
                
                if metric == 'fps':
                    row += f" & ${mean:.1f}$"
                else:
                    row += f" & ${mean:.3f}_{{[{ci_lower:.3f}, {ci_upper:.3f}]}}$"
            
            row += " \\\\\n"
            latex_table += row
        
        latex_table += "\\hline\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        
        with open(f'{save_path}/results_table.tex', 'w') as f:
            f.write(latex_table)
        
        print(f"\nLaTeX table saved to {save_path}/results_table.tex")


def main():
    # Model paths from your training
    model_paths = {
        "YOLOv8": "runs/detect/train94/weights/best.pt",
        "YOLOv9": "runs/detect/train92/weights/best.pt",
        "YOLOv11": "runs/detect/train96/weights/best.pt"
    }
    
    # Dataset path
    dataset_path = "datasets/YOLO_balanced"
    
    # Initialize evaluator
    evaluator = BootstrapEvaluator(
        model_paths=model_paths,
        dataset_path=dataset_path,
        n_bootstrap=1000,  # You can reduce this for faster testing (e.g., 100)
        confidence_level=0.95
    )
    
    # Run bootstrap evaluation
    evaluator.run_bootstrap()
    
    # Calculate statistics
    print("\nCalculating bootstrap statistics...")
    statistics = evaluator.calculate_statistics()
    
    # Perform statistical tests
    print("Performing statistical tests...")
    test_results = evaluator.perform_statistical_tests(statistics)
    
    # Plot results
    print("Creating visualizations...")
    evaluator.plot_results(statistics)
    
    # Save all results
    print("Saving results...")
    evaluator.save_results(statistics, test_results)
    
    # Print summary
    print("\n" + "="*50)
    print("BOOTSTRAP EVALUATION SUMMARY")
    print("="*50)
    
    for model_name in model_paths.keys():
        print(f"\n{model_name}:")
        for metric in ['mAP50-95', 'mAP50', 'precision', 'recall', 'fps']:
            mean = statistics[model_name][metric]['mean']
            ci_lower = statistics[model_name][metric]['ci_lower']
            ci_upper = statistics[model_name][metric]['ci_upper']
            print(f"  {metric}: {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Print significant differences
    print("\n" + "="*50)
    print("STATISTICAL SIGNIFICANCE (p < 0.05)")
    print("="*50)
    
    for metric in ['mAP50-95', 'mAP50', 'precision', 'recall']:
        print(f"\n{metric}:")
        for comparison, results in test_results[metric].items():
            if results['significant']:
                print(f"  {comparison}: p={results['p_value']:.4f}, Cohen's d={results['cohen_d']:.3f}")
    
    print("\nBootstrap evaluation completed!")
    print(f"Results saved in 'bootstrap_results' directory")


if __name__ == "__main__":
    main()
