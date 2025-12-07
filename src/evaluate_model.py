import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import json
import time


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    if union <= 0:
        return 0.0
    return inter / union


def evaluate_at_threshold(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Evaluate predictions at a specific IoU and score threshold.
    Returns precision, recall, and AP.
    """
    all_detections = []
    all_ground_truths = []
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()
        
        # Filter by score threshold
        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        pred_labels = pred_labels[mask]
        
        all_detections.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels
        })
        
        all_ground_truths.append({
            'boxes': gt_boxes,
            'labels': gt_labels
        })
    
    # Compute metrics
    total_tp = 0
    total_fp = 0
    total_gt = sum(len(gt['boxes']) for gt in all_ground_truths)
    
    for det, gt in zip(all_detections, all_ground_truths):
        matched_gt = set()
        
        for i, det_box in enumerate(det['boxes']):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt['boxes']):
                if j in matched_gt:
                    continue
                    
                iou_val = compute_iou(det_box, gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    
    return precision, recall


def evaluate_model_comprehensive(model, data_loader, device, save_dir="evaluation_results", model_name="RCNN"):
    """
    Comprehensive evaluation of object detection model.
    Creates visualizations similar to the paper figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Storage for predictions and targets
    all_predictions = []
    all_targets = []
    all_images = []
    
    # Timing
    inference_times = []
    
    print(f"Running inference for {model_name}...")
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images_list = [img.to(device) for img in images]
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images_list)
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(images))
            
            all_predictions.extend(outputs)
            all_targets.extend(targets)
            all_images.extend([img.cpu() for img in images])
    
    avg_inference_time = np.mean(inference_times)
    print(f"Average inference time: {avg_inference_time:.4f} seconds per image")
    
    # 1. Performance Comparison at Different IoU Thresholds
    print("Computing performance at different thresholds...")
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    avg_precisions = []
    
    for iou_thr in iou_thresholds:
        precision, recall = evaluate_at_threshold(all_predictions, all_targets, 
                                                   iou_threshold=iou_thr, 
                                                   score_threshold=0.5)
        avg_precisions.append(precision)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(iou_thresholds))
    width = 0.35
    
    plt.bar(x, avg_precisions, width, label=model_name, color='#FF8C00')
    
    plt.xlabel('IoU Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('Average Precision (AP)', fontsize=12, fontweight='bold')
    plt.title(f'Performance Comparison of {model_name}', fontsize=14, fontweight='bold')
    plt.xticks(x, [str(t) for t in iou_thresholds])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(avg_precisions):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_performance_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved performance comparison to {save_dir}")
    
    # 2. Precision-Recall Curves at Different IoU Thresholds
    print("Computing Precision-Recall curves...")
    plt.figure(figsize=(10, 8))
    
    colors = ['#FF1493', '#FF4500', '#32CD32', '#1E90FF', '#9370DB', '#FFD700']
    iou_thresholds_pr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    for idx, iou_thr in enumerate(iou_thresholds_pr):
        precisions = []
        recalls = []
        
        # Vary score threshold to get PR curve
        for score_thr in np.linspace(0.05, 0.95, 20):
            precision, recall = evaluate_at_threshold(all_predictions, all_targets,
                                                       iou_threshold=iou_thr,
                                                       score_threshold=score_thr)
            precisions.append(precision)
            recalls.append(recall)
        
        # Sort by recall
        sorted_pairs = sorted(zip(recalls, precisions))
        recalls_sorted = [r for r, p in sorted_pairs]
        precisions_sorted = [p for r, p in sorted_pairs]
        
        plt.plot(recalls_sorted, precisions_sorted, 
                color=colors[idx % len(colors)], 
                linewidth=2,
                label=f'IAC={idx+1}')
    
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_precision_recall_curve.png'), dpi=300)
    plt.close()
    print(f"Saved precision-recall curve to {save_dir}")
    
    # 3. Visualize Detection Results (Grid of Examples)
    print("Creating detection result visualizations...")
    num_examples = min(10, len(all_images))
    
    # Select diverse examples
    indices = np.linspace(0, len(all_images) - 1, num_examples, dtype=int)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, img_idx in enumerate(indices):
        img = all_images[img_idx]
        pred = all_predictions[img_idx]
        target = all_targets[img_idx]
        
        # Convert tensor to numpy
        img_np = img.permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Convert to BGR for OpenCV
        img_bgr = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        
        # Draw predictions (red boxes)
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        for box, score in zip(pred_boxes, pred_scores):
            if score > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_bgr, f'{score:.2f}', (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw ground truth (blue boxes)
        gt_boxes = target['boxes'].cpu().numpy()
        for box in gt_boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Convert back to RGB for matplotlib
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img_rgb)
        axes[idx].axis('off')
        axes[idx].set_title(f'Example {idx+1}', fontsize=10)
    
    plt.suptitle(f'Detection Results from {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_detection_results.png'), dpi=300)
    plt.close()
    print(f"Saved detection results to {save_dir}")
    
    # 4. Save numerical results
    results_dict = {
        'model_name': model_name,
        'avg_inference_time': float(avg_inference_time),
        'performance_by_iou': {
            str(iou): float(ap) for iou, ap in zip(iou_thresholds, avg_precisions)
        }
    }
    
    with open(os.path.join(save_dir, f'{model_name}_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*50}")
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds per image")
    print(f"\nAverage Precision by IoU Threshold:")
    for iou, ap in zip(iou_thresholds, avg_precisions):
        print(f"  IoU {iou}: {ap:.4f}")
    print(f"{'='*50}\n")
    
    return results_dict


def compare_models(results_list, save_dir="evaluation_results"):
    """
    Compare multiple models and create comparison visualizations.
    
    Args:
        results_list: List of dictionaries containing model evaluation results
    """
    if len(results_list) < 2:
        print("Need at least 2 models to compare")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    model_names = [r['model_name'] for r in results_list]
    iou_thresholds = sorted([float(k) for k in results_list[0]['performance_by_iou'].keys()])
    
    # Create comparison bar chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(iou_thresholds))
    width = 0.35
    
    for i, result in enumerate(results_list):
        aps = [result['performance_by_iou'][str(iou)] for iou in iou_thresholds]
        offset = width * (i - 0.5)
        color = '#FFA500' if i == 0 else '#FF4500'
        plt.bar(x + offset, aps, width, label=result['model_name'], color=color)
    
    plt.xlabel('IoU Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('Average Precision (AP)', fontsize=12, fontweight='bold')
    plt.title('Performance Comparison: Fast RCNN vs RCNN', fontsize=14, fontweight='bold')
    plt.xticks(x, [str(t) for t in iou_thresholds])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'models_comparison.png'), dpi=300)
    plt.close()
    
    print(f"Saved model comparison to {save_dir}")
    
    # Create timing comparison
    plt.figure(figsize=(8, 6))
    times = [r['avg_inference_time'] for r in results_list]
    colors = ['#FFA500', '#FF4500'][:len(results_list)]
    
    plt.bar(model_names, times, color=colors)
    plt.ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Inference Time Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    for i, (name, time_val) in enumerate(zip(model_names, times)):
        plt.text(i, time_val + 0.001, f'{time_val:.4f}s', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'timing_comparison.png'), dpi=300)
    plt.close()
    
    print(f"Saved timing comparison to {save_dir}")
