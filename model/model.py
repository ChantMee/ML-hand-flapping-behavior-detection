import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import positional_encoding
import encoders


class model1(nn.Module):
    def __init__(self, output_dim=2, dropout=0.3):
        super(model1, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forecast(self, x):
        self.eval()
        res = self.forward(x)
        self.train()
        return res


class model2(nn.Module):
    def __init__(self, output_dim=2, dropout=0.3):
        super(model2, self).__init__()
        # resnet18
        self.backbone = models.resnet18(pretrained=True)
        # remove pooling and fc layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # add a conv layer to reduce the channel
        self.backbone[7] = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.positional_embedding = positional_encoding.PositionalEncoding(16 * 7 * 7)
        encoder_layer = nn.TransformerEncoderLayer(d_model=16 * 7 * 7, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32 * 16, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, 3, 112, 112)
        x = self.backbone(x)
        x = x.view(batch_size, -1, 16 * 7 * 7)
        x = self.positional_embedding(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = x.view(batch_size, -1)
        x = self.fc3(x)
        return x

    def forecast(self, x):
        self.eval()
        res = self.forward(x)
        self.train()
        return res


class model3(nn.Module):
    def __init__(self, output_dim=2, dropout=0.3):
        super(model3, self).__init__()
        # resnet18
        self.backbone = models.resnet18(pretrained=True)
        # remove pooling and fc layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # add a conv layer to reduce the channel
        self.backbone[7] = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.positional_embedding = positional_encoding.PositionalEncoding(16 * 7 * 7)
        encoder_layer = nn.TransformerEncoderLayer(d_model=16 * 7 * 7, nhead=4)
        self.encoder = nn.MultiheadAttention(16 * 7 * 7, 4)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32 * 16, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, 3, 112, 112)
        x = self.backbone(x)
        x = x.view(batch_size, -1, 16 * 7 * 7)
        x = self.positional_embedding(x)
        x = x.permute(1, 0, 2)
        x, _ = self.encoder(x, x, x)
        x = x.permute(1, 0, 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = x.view(batch_size, -1)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = model3()
    from pre_process.data import MLFPDataset
    from my_transforms import get_transform
    from torch.utils.data import DataLoader
    from utils import count_parameters

    num_paras = count_parameters(model)
    print(f'The model has {num_paras} trainable parameters')
    dataset = MLFPDataset(
        annotation_dir=r'C:\Users\chant\OneDrive\Courses\WOA7015 Advanced Machine Learning\final\dataset\video_annotation.json',
        frame_dir=r'C:\Users\chant\OneDrive\Courses\WOA7015 Advanced Machine Learning\final\dataset\frames',
        duration=2,
        num_frame=16,
        transform=get_transform(0))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    x, y = next(iter(dataloader))
    y_hat = model(x)
    print(y_hat.shape)
