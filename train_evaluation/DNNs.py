import torch
import torch.nn as nn


class OneDimensionalCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_size):
        super(OneDimensionalCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(out_channels * ((input_size - kernel_size + 1) // 2), feature_size)
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, list_length=None):
        x = x.to(torch.float32)
        x1 = self.conv(x)
        # remove the effect of the padding
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, :, list_length[item_idx]:] = 0
        x1 = self.relu(x1)
        x1 = self.adaptive_pool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)
        return x1


class GRUNet(nn.Module):
    def __init__(self, hidden_size, num_features, is_bidirectional=False):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True,
                          bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional

    def forward(self, x):
        x = x.to(torch.float32)
        _, h_n = self.gru(x)
        if self.is_bidirectional:
            return h_n.mean(0)
        return h_n.squeeze(0)

