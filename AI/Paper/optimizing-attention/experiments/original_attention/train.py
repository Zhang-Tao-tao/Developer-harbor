NUM_EPOCHS = 50
BATCH_SIZE = 128
LR = 0.001

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from model import Net
from icecream import ic
from tqdm import tqdm
import os
import shutil
import json

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ic(f"Using device: {device}")
# load dataset
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    root="../data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="../data", train=False, download=True, transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# define model
model = Net(
    image_size=(28, 28),
    patch_size=(4, 4),
    num_classes=10,
    dim=64,
    channels=1,
).to(device)

# define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def accuracy(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / len(target)


# training loop
step_log = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
epoch_log = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs", leave=True):
    model.train()
    temp_log = {"train_loss": [], "train_acc": []}
    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc="Training", leave=False
    ):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        acc = accuracy(output, target)

        step_log["train_loss"].append(loss.item())
        step_log["train_acc"].append(acc)
        temp_log["train_loss"].append(loss.item())
        temp_log["train_acc"].append(acc)
    epoch_log["train_loss"].append(
        sum(temp_log["train_loss"]) / len(temp_log["train_loss"])
    )
    epoch_log["train_acc"].append(
        sum(temp_log["train_acc"]) / len(temp_log["train_acc"])
    )
    ic(
        f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_log["train_loss"][-1]:.4f}, Train Acc: {epoch_log["train_acc"][-1]:.4f}'
    )

    model.eval()
    temp_log = {"test_loss": [], "test_acc": []}
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(test_loader), leave=False, total=len(test_loader), desc="Testing"
        ):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            acc = accuracy(output, target)

            step_log["test_loss"].append(loss.item())
            step_log["test_acc"].append(acc)
            temp_log["test_loss"].append(loss.item())
            temp_log["test_acc"].append(acc)
    epoch_log["test_loss"].append(
        sum(temp_log["test_loss"]) / len(temp_log["test_loss"])
    )
    epoch_log["test_acc"].append(sum(temp_log["test_acc"]) / len(temp_log["test_acc"]))
    ic(
        f'Epoch {epoch+1}/{NUM_EPOCHS}, Test Loss: {epoch_log["test_loss"][-1]:.4f}, Test Acc: {epoch_log["test_acc"][-1]:.4f}'
    )

    # save model
    if epoch == 0:
        if os.path.exists("./saved_model"):
            shutil.rmtree("./saved_model")
        os.makedirs("./saved_model", exist_ok=True)
    torch.save(model.state_dict(), f"./saved_model/model_{epoch}.pth")

json.dump(epoch_log, open("./epoch_log.json", "w+"))
json.dump(step_log, open("./step_log.json", "w+"))
