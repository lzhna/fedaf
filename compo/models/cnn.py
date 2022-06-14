from torch import nn
import torch
from torch.nn import functional as F
from copy import deepcopy

class CNNCifar(nn.Module):
    def __init__(self, num_layers, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        
        self.hidden = nn.ModuleList()
        for k in range(num_layers):
            self.hidden.append(nn.Conv2d(64, 64, 3, padding=1))
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, feats_also=False):
        x = self.conv1(x)           # 32*30*30
        x = self.pool(F.relu(x))    # 32*15*15
        x = self.conv2(x)           # 32*13*13
        x = self.pool(F.relu(x))    # 64*6*6
        x = self.conv3(x)           # 64*4*4
        x = F.relu(x)
        
        for l in self.hidden:
            x = F.relu(l(x))
        feat = x.view(-1, 1024)
        x = self.fc1(feat)
        x = F.relu(x)
        x = self.fc2(x)

        if feats_also:
            return feat, x
        else:
            return x

if __name__ == '__main__':
    def get_n_model_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    data = torch.randn(2,3,32,32)
    net = CNNCifar(0, 10)
    print(f"CNNCifar has n_params={get_n_model_params(net)}M.")

    # p1 = net(data)
    # print(f'p1={p1}')
    # params = deepcopy(net).state_dict()
    # for key in params.keys():
    #     params[key] /= 2.0
    # p2 = net(data)
    # print(f'p2={p2}')