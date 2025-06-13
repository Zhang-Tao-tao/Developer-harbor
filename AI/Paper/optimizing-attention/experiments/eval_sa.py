LOAD_MODEL_NUM_EPOCHS = 49

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from model import OptimAttn
from icecream import ic
from tqdm import tqdm
import pandas as pd
import os
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
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False
)

# define model
model = OptimAttn(
    image_size=(28, 28),
    patch_size=(4, 4),
    num_classes=10,
    dim=64,
    channels=1,
    solver='kaiwu_sa',
    args_solver={
        'user_id': '69878024601862146',
        'sdk_code': "0i4T6LY1XygfwN3MWa8Fjq27OaT0sq",
        'is_check': False,
        'num_process': 25,
    },
).to(device)
model.load_state_dict(torch.load(f'./saved_model/model_{LOAD_MODEL_NUM_EPOCHS}.pth', map_location=device, weights_only=True), strict=True)

# training loop
step_log = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
epoch_log = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
model.eval()
if not os.path.exists('eval_kaiwu_sa_result.csv'):
    df = {'idx': [], 'label': [], 'pred': [], 'correct': []}
    for i in range(10):
        df[f'logits_{i}'] = []
    df = pd.DataFrame(df)
    df.to_csv(f'eval_kaiwu_sa_result.csv', index=False)

df = pd.read_csv(f'eval_kaiwu_sa_result.csv')
current_idx = df['idx'].max() if not df.empty else -1

with torch.no_grad():
    for batch_idx, (data, target) in tqdm(
        enumerate(test_loader), leave=False, total=len(test_loader), desc="Testing"
    ):
        if batch_idx <= current_idx:
            continue
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        temp = {
            'idx': batch_idx,
            'label': target.item(),
            'pred': output.argmax(dim=1).item(),
            'correct': (output.argmax(dim=1) == target).item()
        }
        temp.update({f'logits_{i}': output[0, i].item() for i in range(10)})
        df = df.append(temp, ignore_index=True)
        df.to_csv(f'eval_kaiwu_sa_result.csv', index=False)

average_acc = df['correct'].mean()

ic(f"Average test accuracy: {average_acc:.4f}")
