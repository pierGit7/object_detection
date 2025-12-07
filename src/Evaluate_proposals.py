import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt


#########################################
# Load Pascal VOC annotation
#########################################
def load_pascal_voc(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text
    boxes = []

    for obj in root.iter("object"):
        xmin = int(obj.find("bndbox/xmin").text)
        ymin = int(obj.find("bndbox/ymin").text)
        xmax = int(obj.find("bndbox/xmax").text)
        ymax = int(obj.find("bndbox/ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    return filename, boxes


#########################################
# IoU function
#########################################
def iou(box1, box2):
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


#########################################
# Label proposals
#########################################
# Label proposals
#########################################
def label_proposals(proposals, gt_boxes, iou_threshold=0.5):
    """
    Label proposals as true positives (1) or false positives (0) based on IoU with ground truth boxes.
    """
    labels = []
    for p in proposals:
        if any(iou(p, gt) >= iou_threshold for gt in gt_boxes):
            labels.append(1)  # True positive
        else:
            labels.append(0)  # False positive
    return labels




def save_proposals_with_labels(output_file, proposal_dict, k_max):
    """
    Save labeled proposals in the format:
    {
        "image_name": [
            [xmin, ymin, xmax, ymax, label],
            ...
        ],
        ...
    }
    """
    positive_label = 0
    negative_label = 0
    negative_label_thr = 0.75
    positive_label_thr = 0.25
    
    all_labeled = {}
    for image_id in proposal_dict.keys():
        print(f"Processing {image_id}...")
        # Get proposals for this image
        proposals = proposal_dict.get(image_id, [])
        image_id_path = image_id.replace(".jpg", "").replace(".png", "")
        xml_path = os.path.join(ann_dir, image_id_path + ".xml")
        _, gt_boxes = load_pascal_voc(xml_path)
        # Save only the k_max proposals with labels for each image
        labels = label_proposals(proposals, gt_boxes, iou_threshold=0.5)

        # Append labels to proposals
        labeled_props = [prop + [label] for prop, label in zip(proposals, labels)]
        
        balanced_labeled_props = []
        # print(len(labeled_props))
        pos_numbers = positive_label_thr * k_max
        neg_numbers = negative_label_thr * k_max
        for props in labeled_props:
            if props[-1] == 1 and pos_numbers > 0:
                balanced_labeled_props.append(props)
                pos_numbers -= 1
            elif props[-1] == 0 and neg_numbers > 0:
                balanced_labeled_props.append(props)
                neg_numbers -= 1


        # Add/update the current image
        all_labeled[image_id] = balanced_labeled_props

    # Write JSON back
    with open(output_file, "w") as f:
        json.dump(all_labeled, f, indent=2)
    # with open(output_file, "r") as f:
    #     json_data = json.load(f)
    #     print(f"json keys: {len(json_data.keys())}")
    #     for image_id, props in json_data.items():
    #         print(f"Image: {image_id}, Number of proposals: {len(props)}")
        


#########################################
# Main evaluation
#########################################
data_root = "/dtu/datasets1/02516/potholes/"
ann_dir = os.path.join(data_root, "annotations")
def main():


    proposals_json = "/zhome/48/a/213648/work/pier/object_recognition/proposal/selectivesearch_proposals/proposals.json"

    # Load proposals
    with open(proposals_json, "r") as f:
        proposals_dict = json.load(f)
        
    print(f"Loaded proposals for {len(proposals_dict.keys())} images.")
    for image_id, props in proposals_dict.items():
        print(f"Image: {image_id}, Number of proposals: {len(props)}")
    # ==== NEW: No splits.json → Use all annotations ====
    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".xml")])
    train_ids = [os.path.splitext(f)[0] for f in ann_files]

    print(f"Found {len(train_ids)} images (using full dataset as training set).")

    # Proposal counts to test
    K_values = [50, 100, 200, 500, 1000, 1500, 2000]
    iou_threshold = 0.5

    total_gt = 0
    covered_gt_per_K = {K: 0 for K in K_values}

    for img_id in train_ids:
        img_name_jpg = img_id + ".jpg"
        img_name_png = img_id + ".png"

        # Try both
        if img_name_jpg in proposals_dict:
            img_name = img_name_jpg
        elif img_name_png in proposals_dict:
            img_name = img_name_png
        else:
            print(f"No proposals for {img_id}, skipping.")
            continue

        xml_path = os.path.join(ann_dir, img_id + ".xml")
        _, gt_boxes = load_pascal_voc(xml_path)

        total_gt += len(gt_boxes)

        proposals = proposals_dict[img_name]

        for K in K_values:
            curr_props = proposals[:K]

            for gt in gt_boxes:
                hit = any(iou(p, gt) >= iou_threshold for p in curr_props)
                if hit:
                    covered_gt_per_K[K] += 1

    recalls = {K: covered_gt_per_K[K] / total_gt for K in K_values}
    
    K_list = list(recalls.keys())
    R_list = [recalls[K] for K in K_list]
    
    idx_max = np.argmax(R_list)
    k_max = K_list[idx_max]
    print(f"\nMaximum recall of {R_list[idx_max]:.3f} at K={k_max}")
    
    save_proposals_with_labels(
        "/zhome/48/a/213648/work/pier/object_recognition/proposal/proposal_label.json",
        proposals_dict,
        k_max
    )

    print("\nRecall results (IoU ≥ 0.5):")
    for K in K_values:
        print(f"K={K}: recall={recalls[K]:.3f}")
    # --- Create and save recall plot ---
    import matplotlib.pyplot as plt



    plt.figure(figsize=(6, 4))
    plt.plot(K_list, R_list, marker="o", linewidth=2)
    plt.xlabel("Number of proposals per image (K)")
    plt.ylabel("Recall (IoU ≥ 0.5)")
    plt.title("Proposal Recall vs Number of Proposals")
    plt.grid(True)

    out_fig = "/zhome/48/a/213648/work/pier/object_recognition/proposal/proposal_recall_curve.png"
    plt.savefig(out_fig, bbox_inches="tight", dpi=150)

    print(f"\nSaved recall plot to: {out_fig}")


if __name__ == "__main__":
    main()
