"""
Visualization utilities for object detection results.
Can be used independently to visualize predictions.

Usage:
    from visualize_detections import plot_detection, plot_grid_detections
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch


def plot_detection(image, predictions, ground_truth=None, save_path=None, 
                   conf_threshold=0.5, figsize=(12, 8)):
    """
    Plot a single image with predicted and ground-truth boxes.
    
    Args:
        image: Tensor or numpy array [C, H, W] or [H, W, C]
        predictions: Dict with 'boxes', 'scores', 'labels'
        ground_truth: Optional dict with 'boxes', 'labels'
        save_path: Path to save the figure
        conf_threshold: Minimum confidence score to display
        figsize: Figure size
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(image):
        img_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image
    
    # Denormalize if needed (assuming ImageNet normalization)
    if img_np.max() <= 1.0:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
    
    # Convert to uint8
    img_bgr = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    
    # Draw predictions (RED)
    pred_boxes = predictions['boxes'].cpu().numpy() if torch.is_tensor(predictions['boxes']) else predictions['boxes']
    pred_scores = predictions['scores'].cpu().numpy() if torch.is_tensor(predictions['scores']) else predictions['scores']
    
    for box, score in zip(pred_boxes, pred_scores):
        if score >= conf_threshold:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f'{score:.2f}'
            cv2.putText(img_bgr, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw ground truth (BLUE)
    if ground_truth is not None:
        gt_boxes = ground_truth['boxes'].cpu().numpy() if torch.is_tensor(ground_truth['boxes']) else ground_truth['boxes']
        for box in gt_boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(img_bgr, 'GT', (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Convert back to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_rgb)
    ax.axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Predictions'),
    ]
    if ground_truth is not None:
        legend_elements.append(Patch(facecolor='blue', label='Ground Truth'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_grid_detections(images, predictions, ground_truths=None, 
                        save_path=None, conf_threshold=0.5, 
                        grid_size=(2, 5), figsize=(20, 8)):
    """
    Plot a grid of detection results.
    
    Args:
        images: List of image tensors
        predictions: List of prediction dicts
        ground_truths: Optional list of ground truth dicts
        save_path: Path to save the figure
        conf_threshold: Minimum confidence score to display
        grid_size: (rows, cols) for the grid
        figsize: Figure size
    """
    rows, cols = grid_size
    num_images = min(len(images), rows * cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for idx in range(num_images):
        img = images[idx]
        pred = predictions[idx]
        gt = ground_truths[idx] if ground_truths else None
        ax = axes[idx]
        
        # Convert tensor to numpy
        if torch.is_tensor(img):
            img_np = img.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img
        
        # Denormalize
        if img_np.max() <= 1.0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
        
        # Convert to BGR for OpenCV
        img_bgr = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        
        # Draw predictions (RED)
        pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
        pred_scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']
        
        for box, score in zip(pred_boxes, pred_scores):
            if score >= conf_threshold:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_bgr, f'{score:.2f}', (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw ground truth (BLUE)
        if gt is not None:
            gt_boxes = gt['boxes'].cpu().numpy() if torch.is_tensor(gt['boxes']) else gt['boxes']
            for box in gt_boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img_rgb)
        ax.axis('off')
        ax.set_title(f'Example {idx+1}', fontsize=10)
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Detection Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved grid to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path=None, figsize=(12, 6)):
    """
    Plot comparison of metrics across models.
    
    Args:
        metrics_dict: Dict mapping model names to their metrics
                     e.g., {'Fast RCNN': {'AP@0.5': 0.85, ...}, ...}
        save_path: Path to save figure
        figsize: Figure size
    """
    model_names = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#FFA500', '#FF4500', '#32CD32', '#1E90FF']
    
    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        values = [metrics[m] for m in metric_names]
        offset = width * (idx - len(model_names)/2 + 0.5)
        ax.bar(x + offset, values, width, label=model_name, color=colors[idx % len(colors)])
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_inference_time(model_times, save_path=None, figsize=(10, 6)):
    """
    Plot inference time comparison.
    
    Args:
        model_times: Dict mapping model names to inference times
        save_path: Path to save figure
        figsize: Figure size
    """
    models = list(model_times.keys())
    times = list(model_times.values())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#FFA500', '#FF4500']
    bars = ax.bar(models, times, color=colors[:len(models)])
    
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.4f}s',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved timing comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


# Example usage
if __name__ == "__main__":
    # This is just for demonstration
    print("Visualization utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_detection(): Plot single detection result")
    print("  - plot_grid_detections(): Plot grid of detection results")
    print("  - plot_metrics_comparison(): Compare metrics across models")
    print("  - plot_inference_time(): Compare inference times")
