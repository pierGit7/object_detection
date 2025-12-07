import os
import cv2
import xml.etree.ElementTree as ET
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
# Draw bounding boxes
#########################################
def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for (xmin, ymin, xmax, ymax) in boxes:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    return image


#########################################
# Main execution — generates 2×2 grids
#########################################
def main():

    # Data path (DTU dataset)
    data_root = "/dtu/datasets1/02516/potholes/"
    img_dir = os.path.join(data_root, "images")
    ann_dir = os.path.join(data_root, "annotations")

    # Save in your ZHOME
    output_dir = "/zhome/48/a/213648/work/pier/object_recognition/images/selectivesearch_proposals"
    os.makedirs(output_dir, exist_ok=True)

    # How many examples to grab total
    N = 8

    xml_files = [f for f in os.listdir(ann_dir) if f.endswith(".xml")]
    xml_files = xml_files[:N]

    print(f"Creating {(len(xml_files)+3)//4} grids of 2x2 examples...")

    # Process in groups of 4
    for i in range(0, len(xml_files), 4):
        batch = xml_files[i:i+4]

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()

        for ax, xml_file in zip(axs, batch):
            xml_path = os.path.join(ann_dir, xml_file)

            # Load annotation
            filename, boxes = load_pascal_voc(xml_path)
            img_path = os.path.join(img_dir, filename)

            # Skip missing images
            if not os.path.exists(img_path):
                ax.set_title("Missing image")
                ax.axis('off')
                continue

            img = cv2.imread(img_path)
            if img is None:
                ax.set_title("Could not read")
                ax.axis('off')
                continue

            # Draw boxes
            img_drawn = draw_boxes(img.copy(), boxes)

            # Convert BGR → RGB for plotting
            img_rgb = cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB)

            ax.imshow(img_rgb)
            ax.set_title(filename)
            ax.axis("off")

        # Fill empty slots if batch < 4
        for j in range(len(batch), 4):
            axs[j].axis("off")

        # Save the final grid
        grid_name = f"grid_{i//4 + 1}.png"
        save_path = os.path.join(output_dir, grid_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved grid: {save_path}")

    print("Done!")


if __name__ == "__main__":
    main()
