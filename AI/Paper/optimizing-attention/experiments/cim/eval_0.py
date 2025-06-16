IMG_IDX = 22
LOAD_MODEL_NUM_EPOCHS = 49

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from model_0 import OptimAttn
from icecream import ic
from tqdm import tqdm
import pandas as pd
import os
from matplotlib import pyplot as plt
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
    root="../data", train=False, download=True, transform=transform
)

img, label = test_dataset[IMG_IDX]
plt.imshow(img.squeeze().numpy(), cmap='gray')
plt.axis('off')
plt.savefig('eval_img.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.close()

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
    },
).to(device)
model.load_state_dict(torch.load(f'../saved_model/model_{LOAD_MODEL_NUM_EPOCHS}.pth', map_location=device, weights_only=True), strict=True)

    
model.eval()
with torch.no_grad():
    data = img.to(device).unsqueeze(1)
    output = model(data)
    