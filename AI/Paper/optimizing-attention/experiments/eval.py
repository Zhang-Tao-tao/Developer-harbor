LOAD_MODEL_NUM_EPOCHS = 49
BATCH_SIZE = 128

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from model import OptimAttn
from icecream import ic
from tqdm import tqdm
import os
import shutil
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ic(f"Using device: {device}")
# load dataset
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# define model
model = OptimAttn(
    image_size=(28, 28),
    patch_size=(4, 4),
    num_classes=10,
    dim=64,
    channels=1,
    dropout=0.0,
    emb_dropout=0.0
).to(device)
model.load_state_dict(torch.load(f'./saved_model/model_{LOAD_MODEL_NUM_EPOCHS}.pth', map_location=device), strict=True, weight_only=True)

# define accuracy
def accuracy(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / len(target)

# training loop
step_log = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
epoch_log = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
model.eval()
acc_log = []
with torch.no_grad():
    for batch_idx, (data, target) in tqdm(
        enumerate(test_loader), leave=False, total=len(test_loader), desc="Testing"
    ):
        data, target = data.to(device), target.to(device)
        output = model(data)
        acc = accuracy(output, target)

        step_log["test_acc"].extend(acc)

average_acc = sum(step_log["test_acc"]) / len(step_log["test_acc"])

ic(f"Average test accuracy: {average_acc:.4f}")

json.dump(step_log, open(f"eval_result.json", "w+"), indent=4)
