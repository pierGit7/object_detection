#!/usr/bin/env python3
import os
import cv2
import json
import selectivesearch
from tqdm import tqdm


#########################################
# Resize helper
#########################################
def resize_image(img, max_side=600):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0, 1.0

    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    return resized, scale, scale


#########################################
# Main extraction pipeline
#########################################
def main():

    # Dataset path
    data_root = "/dtu/datasets1/02516/potholes/"
    img_dir = os.path.join(data_root, "images")

    # Output
    output_dir = "/zhome/48/a/213648/work/pier/object_recognition/proposal/selectivesearch_proposals"
    os.makedirs(output_dir, exist_ok=True)

    out_json = os.path.join(output_dir, "proposals.json")

    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
    all_proposals = {}

    print(f"Extracting Selective Search proposals for {len(image_files)} images...")

    for img_name in tqdm(image_files):

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read: {img_path}")
            continue

        # Resize for speed
        img_small, sx, sy = resize_image(img)

        # Run Selective Search (fast mode)
        _, regions = selectivesearch.selective_search(
            img_small,
            scale=250,
            sigma=0.9,
            min_size=30
        )

        proposals = []
        seen = set()

        for r in regions:
            x, y, w, h = r["rect"]

            # Skip duplicates
            if (x, y, w, h) in seen:
                continue
            seen.add((x, y, w, h))

            # Convert to original resolution
            xmin = int(x / sx)
            ymin = int(y / sy)
            xmax = int((x + w) / sx)
            ymax = int((y + h) / sy)

            proposals.append([xmin, ymin, xmax, ymax])

        all_proposals[img_name] = proposals

    # Save to JSON
    with open(out_json, "w") as f:
        json.dump(all_proposals, f, indent=2)

    print(f"Done! Saved proposals to: {out_json}")


if __name__ == "__main__":
    main()
