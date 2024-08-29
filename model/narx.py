import torch.nn as nn
from torch.nn.functional import leaky_relu, relu


class NarxModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.linear1 = nn.Linear(73, 100)
        self.linear2 = nn.Linear(100, 7)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, batch):
        batch = relu(self.linear1(batch))
        batch = self.dropout(batch)
        batch = leaky_relu(self.linear2(batch))
        return batch