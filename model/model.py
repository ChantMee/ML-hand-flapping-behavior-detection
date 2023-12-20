import torch
import torch.nn as nn
import torch.nn.functional as F


class MLFP(nn.Module):
    def __init__(self, output_dim=2):
        super(MLFP, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 15 * 15, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))  # 4 * 15 * 15
        x = x.view(-1, 64 * 4 * 15 * 15)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forecast(self, x):
        self.eval()
        res = self.forward(x)
        self.train()
        return res