"""
Example script to run comprehensive model evaluation.
This script demonstrates how to evaluate R-CNN and Fast R-CNN models
and create visualizations similar to the research paper figure.

Usage:
    python run_evaluation.py
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataset_pothole import PotholeProposalsDataset
from model.rcnn import RCNN_VGG16
from evaluate_model import evaluate_model_comprehensive, compare_models


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths (modify these to your paths)
    img_dir = "/dtu/datasets1/02516/potholes/images"
    proposals_json = "/zhome/48/a/213648/work/pier/object_recognition/proposal/proposal_label.json"
    
    # Transforms
    test_transforms = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    dataset_test = PotholeProposalsDataset(
        img_dir,
        proposals_json,
        transform=test_transforms,
        split="test"
    )
    
    test_loader = DataLoader(
        dataset_test, 
        batch_size=8, 
        shuffle=False, 
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    print(f"Test dataset size: {len(dataset_test)}")
    
    # ========================================
    # Evaluate R-CNN Model
    # ========================================
    print("\n" + "="*60)
    print("EVALUATING R-CNN MODEL")
    print("="*60)
    
    # Load your trained R-CNN model
    rcnn_model = RCNN_VGG16(num_classes=2)
    
    # Load trained weights (modify path to your checkpoint)
    try:
        checkpoint_path = "path/to/your/rcnn_checkpoint.pth"
        rcnn_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded R-CNN checkpoint from {checkpoint_path}")
    except:
        print("Warning: Could not load R-CNN checkpoint. Using random weights.")
    
    rcnn_model = rcnn_model.to(device)
    
    # Run comprehensive evaluation
    rcnn_results = evaluate_model_comprehensive(
        model=rcnn_model,
        data_loader=test_loader,
        device=device,
        save_dir="evaluation_results/rcnn",
        model_name="RCNN"
    )
    
    # ========================================
    # Evaluate Fast R-CNN Model
    # ========================================
    print("\n" + "="*60)
    print("EVALUATING FAST R-CNN MODEL")
    print("="*60)
    
    # Load Fast R-CNN (Faster R-CNN from torchvision)
    fast_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    
    # Load trained weights (modify path to your checkpoint)
    try:
        checkpoint_path = "path/to/your/fast_rcnn_checkpoint.pth"
        fast_rcnn_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded Fast R-CNN checkpoint from {checkpoint_path}")
    except:
        print("Warning: Could not load Fast R-CNN checkpoint. Using random weights.")
    
    fast_rcnn_model = fast_rcnn_model.to(device)
    
    # Run comprehensive evaluation
    fast_rcnn_results = evaluate_model_comprehensive(
        model=fast_rcnn_model,
        data_loader=test_loader,
        device=device,
        save_dir="evaluation_results/fast_rcnn",
        model_name="Fast RCNN"
    )
    
    # ========================================
    # Compare Both Models
    # ========================================
    print("\n" + "="*60)
    print("COMPARING MODELS")
    print("="*60)
    
    compare_models(
        results_list=[fast_rcnn_results, rcnn_results],
        save_dir="evaluation_results/comparison"
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("Results saved in:")
    print("  - evaluation_results/rcnn/")
    print("  - evaluation_results/fast_rcnn/")
    print("  - evaluation_results/comparison/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
