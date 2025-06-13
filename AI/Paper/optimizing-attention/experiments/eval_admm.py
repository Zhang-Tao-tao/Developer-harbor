LOAD_MODEL_NUM_EPOCHS = 49
BATCH_SIZE = 128

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from model import OptimAttn
from icecream import ic
from tqdm import tqdm
import pandas as pd
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
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# define model
model = OptimAttn(
    image_size=(28, 28),
    patch_size=(4, 4),
    num_classes=10,
    dim=64,
    channels=1,
).to(device)
model.load_state_dict(torch.load(f'./saved_model/model_{LOAD_MODEL_NUM_EPOCHS}.pth', map_location=device, weights_only=True), strict=True)
    
model.eval()
df = {'label': [], 'pred': [], 'correct': []}
for i in range(10):
    df[f'logits_{i}'] = []
with torch.no_grad():
    for batch_idx, (data, target) in tqdm(
        enumerate(test_loader), leave=False, total=len(test_loader), desc="Testing"
    ):
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        df['label'].extend(target.cpu().numpy().tolist())
        df['pred'].extend(output.argmax(dim=1).cpu().numpy().tolist())
        for i in range(10):
            df[f'logits_{i}'].extend(output[:, i].cpu().numpy().tolist())
        df['correct'].extend((output.argmax(dim=1) == target).cpu().numpy().tolist())

df = pd.DataFrame(df)
df.to_csv(f'eval_admm_result.csv', index=False)

average_acc = df['correct'].mean()

ic(f"Average test accuracy: {average_acc:.4f}")
