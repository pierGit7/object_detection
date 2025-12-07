import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_pothole import PotholeProposalsDataset
from torchvision.models import resnet18
from train import train_object_detection
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn


train_transforms = transforms.Compose([
    transforms.Resize((250, 250)),            # resize crop
    transforms.RandomHorizontalFlip(p=0.5),   # augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
img_dir = "/dtu/datasets1/02516/potholes/images"
proposals_json = "/zhome/48/a/213648/work/pier/object_recognition/proposal/proposal_label.json"

dataset_train = PotholeProposalsDataset(
    img_dir,
    proposals_json,
    transform=train_transforms,
    split="train"
)
dataset_test = PotholeProposalsDataset(
    img_dir,
    proposals_json,
    transform=train_transforms,
    split="test"
)
dataset_valid = PotholeProposalsDataset(
    img_dir,
    proposals_json,
    transform=train_transforms,
    split="valid"
)

# create dataloaders
train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=4)
valid_loader = DataLoader(dataset_valid, batch_size=32, shuffle=False, num_workers=4)

# # Load pretrained ResNet-18
# model = resnet18(pretrained=True)

# # Replace the final fully-connected layer for 3 classes
# in_features = model.fc.in_features
# model.fc = nn.Linear(in_features, 3) #n +1 classes (pothole + non-pothole + background)

model = fasterrcnn_resnet50_fpn(pretrained=True)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_object_detection(
    model,
    train_loader,
    valid_loader,
    test_loader,
    dataset_train,
    dataset_valid,
    dataset_test,
    optimizer,
    num_epochs=10
)
