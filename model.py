import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from torch import device
from torch.utils.data import Dataset, DataLoader

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

IMG_SIZE = 300
CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

class ID_Dataset(Dataset):

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_src = self.df.loc[idx, 'full_path']
        # print(image_src)
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = self.df.loc[idx, CATEGORIES].values
        labels = torch.from_numpy(labels.astype(np.int8))
        labels = labels.unsqueeze(-1)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, labels


class ID_Model(nn.Module):

    def __init__(self, num_classes=4, initfc_type='normal', gain=0.2):
        super().__init__()
        model = EfficientNet.from_pretrained('efficientnet-b3')

        self.model = model
        self.fc = nn.Linear(model._conv_head.out_channels, num_classes)

        if hasattr(self.fc, 'bias') and self.fc.bias is not None:
            nn.init.constant_(self.fc.bias.data, 0.0)
        if initfc_type == 'normal':
            nn.init.normal_(self.fc.weight.data, 0.0, gain)
        elif initfc_type == 'xavier':
            nn.init.xavier_normal_(self.fc.weight.data, gain=gain)
        elif initfc_type == 'kaiming':
            nn.init.kaiming_normal_(self.fc.weight.data, a=0, mode='fan_in')
        elif initfc_type == 'orthogonal':
            nn.init.orthogonal_(self.fc.weight.data, gain=gain)

    def forward(self, x):
        x = self.model.extract_features(x)
        x = x * torch.sigmoid(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


temp_df = pd.DataFrame([[os.path.join(__location__, 'temp.jpg'), 0, 0, 0, 0, 0, 0]], columns=['full_path'] + CATEGORIES)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ID_Model(num_classes=len(CATEGORIES))
model.load_state_dict(torch.load(os.path.join(__location__, 'model.pth'), map_location=device))
model = model.to(device)
model.eval()
print("Model Loading Completed")

def predictions(img):
    transforms_preds = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ])
    image = img
    image.save(os.path.join(__location__, 'temp.jpg'))
    dataset_test = ID_Dataset(df=temp_df, transforms=transforms_preds)
    dataloader_preds = DataLoader(dataset_test, batch_size=1, shuffle=False)

    for step, batch in enumerate(dataloader_preds):
        images = batch[0]
        images = images.to(device, dtype=torch.float)
        with torch.no_grad():
            outputs = model(images)

            test_preds = outputs.data.cpu()

    s = ('-----\n')
    d = {}
    for idx in torch.topk(outputs, k=6).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        s = s + ('{label}{space}({p:.2f}%)\n'.format(label=CATEGORIES[idx], space=' ' * (20 - len(CATEGORIES[idx])),
                                                     p=prob * 100))
        d[CATEGORIES[idx]] = prob
    return s
