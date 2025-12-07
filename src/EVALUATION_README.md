# Object Detection Evaluation Framework

This evaluation framework provides comprehensive analysis tools for object detection models, similar to academic paper visualizations.

## Features

The evaluation framework generates three main types of visualizations:

### 1. Performance Comparison Chart
- Bar chart showing Average Precision (AP) at different IoU thresholds
- Compares model performance across various IoU thresholds (0.1 to 0.8)
- Helps identify optimal detection thresholds

### 2. Precision-Recall Curves
- Multiple curves for different IoU thresholds (IAC=1 to IAC=6)
- Shows the trade-off between precision and recall
- Useful for understanding model behavior at different confidence levels

### 3. Detection Results Visualization
- Grid of example images with predicted and ground-truth bounding boxes
- Predicted boxes shown in RED with confidence scores
- Ground-truth boxes shown in BLUE
- Helps visually assess detection quality

## Files

- **`evaluate_model.py`**: Main evaluation module with comprehensive metrics
- **`run_evaluation.py`**: Example script demonstrating how to use the evaluation framework

## Usage

### Basic Evaluation (Single Model)

```python
from evaluate_model import evaluate_model_comprehensive

# Evaluate your model
results = evaluate_model_comprehensive(
    model=your_model,
    data_loader=test_loader,
    device=device,
    save_dir="evaluation_results/my_model",
    model_name="My Model"
)
```

### Comparing Multiple Models

```python
from evaluate_model import evaluate_model_comprehensive, compare_models

# Evaluate first model
results1 = evaluate_model_comprehensive(
    model=model1,
    data_loader=test_loader,
    device=device,
    save_dir="evaluation_results/model1",
    model_name="Fast RCNN"
)

# Evaluate second model
results2 = evaluate_model_comprehensive(
    model=model2,
    data_loader=test_loader,
    device=device,
    save_dir="evaluation_results/model2",
    model_name="RCNN"
)

# Compare models
compare_models(
    results_list=[results1, results2],
    save_dir="evaluation_results/comparison"
)
```

### Running the Example Script

```bash
# Make sure you're in the src directory
cd src/

# Run the evaluation script
python run_evaluation.py
```

**Note**: You need to modify the paths in `run_evaluation.py`:
- Update `checkpoint_path` to point to your trained model weights
- Update `img_dir` and `proposals_json` if using different paths

## Output Structure

After running evaluation, you'll get:

```
evaluation_results/
├── rcnn/
│   ├── RCNN_performance_comparison.png      # AP vs IoU threshold
│   ├── RCNN_precision_recall_curve.png      # PR curves
│   ├── RCNN_detection_results.png           # Example detections
│   └── RCNN_results.json                    # Numerical results
├── fast_rcnn/
│   ├── Fast_RCNN_performance_comparison.png
│   ├── Fast_RCNN_precision_recall_curve.png
│   ├── Fast_RCNN_detection_results.png
│   └── Fast_RCNN_results.json
└── comparison/
    ├── models_comparison.png                # Side-by-side AP comparison
    └── timing_comparison.png                # Inference time comparison
```

## Metrics Computed

1. **Average Precision (AP)**: Computed at multiple IoU thresholds
2. **Precision**: True Positives / (True Positives + False Positives)
3. **Recall**: True Positives / Total Ground Truth
4. **Inference Time**: Average time per image in seconds
5. **Precision-Recall Curves**: At various IoU thresholds

## Customization

### Adjusting IoU Thresholds

In `evaluate_model.py`, modify:

```python
iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # For bar chart
iou_thresholds_pr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]         # For PR curves
```

### Adjusting Score Thresholds

Change the confidence threshold for visualizations:

```python
# In evaluate_model_comprehensive function
if score > 0.5:  # Change this value (0.0 to 1.0)
```

### Number of Example Images

Modify the number of detection examples shown:

```python
num_examples = min(10, len(all_images))  # Change 10 to desired number
```

## Requirements

- PyTorch
- torchvision
- numpy
- matplotlib
- opencv-python (cv2)
- PIL
- tqdm
- torchmetrics

## Tips

1. **GPU Acceleration**: The evaluation runs faster on GPU. Make sure CUDA is available.
2. **Batch Size**: Adjust batch size in DataLoader based on your GPU memory.
3. **Score Threshold**: Lower thresholds increase recall but decrease precision.
4. **IoU Threshold**: Higher IoU thresholds require more precise localization.

## Example Results Interpretation

### Performance Comparison Chart
- Higher bars = Better performance
- Compare across different IoU thresholds to see where model excels
- Typically, AP decreases as IoU threshold increases

### Precision-Recall Curves
- Curves closer to top-right corner = Better performance
- Higher curves = Better precision-recall trade-off
- Different colors represent different IoU thresholds

### Detection Results
- Red boxes = Model predictions with confidence scores
- Blue boxes = Ground truth annotations
- Good detections have high overlap between red and blue boxes

## Troubleshooting

**Issue**: "Could not load checkpoint"
- **Solution**: Make sure checkpoint paths are correct in `run_evaluation.py`

**Issue**: "CUDA out of memory"
- **Solution**: Reduce batch size in DataLoader

**Issue**: "No predictions generated"
- **Solution**: Check if model is properly loaded and in eval mode

**Issue**: Missing visualizations
- **Solution**: Ensure `save_dir` has write permissions

## Citation

If you use this evaluation framework in your research, please cite appropriately.
