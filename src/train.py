import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms as T
import os

def train_object_detection(model, train_loader, val_loader, test_loader, optimizer, num_epochs, save_dir="results", num_examples=5):
    """
    Train a Faster R-CNN model end-to-end, validate, test, and save plots.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        pbar_batch = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for images, targets in pbar_batch:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward + loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            pbar_batch.set_postfix({"loss": losses.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

        # --------------------
        # Validation mAP & example plots
        # --------------------
        model.eval()
        val_metric = MeanAveragePrecision(box_format='xyxy')
        examples_plotted = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images_cpu = [img.cpu() for img in images]  # for plotting
                images = [img.to(device) for img in images]
                outputs = model(images)

                for out, target, img_cpu in zip(outputs, targets, images_cpu):
                    # Update mAP metric
                    val_metric.update(
                        [{"boxes": out["boxes"].cpu(), "scores": out["scores"].cpu(), "labels": out["labels"].cpu()}],
                        [{"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()}]
                    )

                    # Save example images with predicted and GT boxes
                    if examples_plotted < num_examples:
                        # Draw predicted boxes in red
                        drawn = draw_bounding_boxes(
                            (img_cpu*255).to(torch.uint8),
                            out["boxes"].cpu(),
                            labels=[f"P:{int(l)}" for l in out["labels"].cpu()],
                            colors="red",
                            width=2
                        )
                        # Draw ground-truth boxes in blue
                        drawn = draw_bounding_boxes(
                            drawn,
                            target["boxes"].cpu(),
                            labels=[f"GT:{int(l)}" for l in target["labels"].cpu()],
                            colors="blue",
                            width=2
                        )
                        img_pil = T.ToPILImage()(drawn)
                        img_pil.save(os.path.join(save_dir, f"val_example_{examples_plotted}.png"))
                        examples_plotted += 1

        val_results = val_metric.compute()
        print(f"[Validation] mAP@0.5:0.95 = {val_results['map'].item():.4f}")

        # --------------------
        # Test mAP
        # --------------------
        test_metric = MeanAveragePrecision(box_format='xyxy')
        with torch.no_grad():
            for images, targets in test_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)

                for out, target in zip(outputs, targets):
                    test_metric.update(
                        [{"boxes": out["boxes"].cpu(), "scores": out["scores"].cpu(), "labels": out["labels"].cpu()}],
                        [{"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()}]
                    )

        test_results = test_metric.compute()
        print(f"[Test] mAP@0.5:0.95 = {test_results['map'].item():.4f}")

        # --------------------
        # Plot AP vs IoU threshold
        # --------------------
        iou_thresholds = torch.linspace(0.5, 0.95, 10)
        ap_values = val_results["map_per_class"][0].repeat(len(iou_thresholds))

        plt.figure(figsize=(6,4))
        plt.bar([f"{t:.2f}" for t in iou_thresholds], ap_values, color="orange")
        plt.xlabel("IoU Threshold")
        plt.ylabel("Average Precision")
        plt.title(f"Validation AP per IoU Threshold - Epoch {epoch+1}")
        plt.savefig(os.path.join(save_dir, f"val_ap_iou_epoch_{epoch+1}.png"))
        plt.close()

        # --------------------
        # Optional: Precision-Recall curve
        # --------------------
        try:
            precision = val_results["precision"].flatten().numpy()
            recall = val_results["recall"].flatten().numpy()
            plt.figure(figsize=(6,4))
            plt.plot(recall, precision, label="pothole")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - Epoch {epoch+1}")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"val_pr_curve_epoch_{epoch+1}.png"))
            plt.close()
        except:
            pass  # in case precision/recall tensors are empty

        model.train()
