"""
Quick evaluation script for a single model.
This is a simplified version for quickly testing a single model.

Usage:
    python quick_eval.py --model_path path/to/model.pth --model_type rcnn
    
Arguments:
    --model_path: Path to model checkpoint
    --model_type: Type of model (rcnn or fast_rcnn)
    --batch_size: Batch size for evaluation (default: 8)
    --save_dir: Directory to save results (default: quick_eval_results)
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataset_pothole import PotholeProposalsDataset
from model.rcnn import RCNN_VGG16
from evaluate_model import evaluate_model_comprehensive


def parse_args():
    parser = argparse.ArgumentParser(description='Quick Model Evaluation')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['rcnn', 'fast_rcnn'],
                       help='Type of model: rcnn or fast_rcnn')
    parser.add_argument('--img_dir', type=str,
                       default='/dtu/datasets1/02516/potholes/images',
                       help='Path to images directory')
    parser.add_argument('--proposals_json', type=str,
                       default='/zhome/48/a/213648/work/pier/object_recognition/proposal/proposal_label.json',
                       help='Path to proposals JSON file')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='quick_eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    
    return parser.parse_args()


def load_model(model_type, checkpoint_path, device):
    """Load model based on type"""
    
    if model_type == 'rcnn':
        print("Loading R-CNN model...")
        model = RCNN_VGG16(num_classes=2)
    elif model_type == 'fast_rcnn':
        print("Loading Fast R-CNN model...")
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully!")
    return model


def main():
    args = parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Transforms
    test_transforms = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    print(f"\nLoading dataset from: {args.img_dir}")
    print(f"Using proposals from: {args.proposals_json}")
    
    dataset_test = PotholeProposalsDataset(
        args.img_dir,
        args.proposals_json,
        transform=test_transforms,
        split="test"
    )
    
    test_loader = DataLoader(
        dataset_test, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    print(f"Test dataset size: {len(dataset_test)} images")
    print(f"Batch size: {args.batch_size}")
    
    # Load model
    model = load_model(args.model_type, args.model_path, device)
    model = model.to(device)
    
    # Model name for display
    model_name = "RCNN" if args.model_type == 'rcnn' else "Fast RCNN"
    
    # Run evaluation
    print("\n" + "="*60)
    print(f"EVALUATING {model_name.upper()} MODEL")
    print("="*60 + "\n")
    
    results = evaluate_model_comprehensive(
        model=model,
        data_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        model_name=model_name
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {args.save_dir}/")
    print("\nGenerated files:")
    print(f"  - {model_name}_performance_comparison.png")
    print(f"  - {model_name}_precision_recall_curve.png")
    print(f"  - {model_name}_detection_results.png")
    print(f"  - {model_name}_results.json")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
