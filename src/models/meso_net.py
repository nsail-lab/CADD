"""
This code is modified version of MesoNet DeepFake detection solution
from FakeAVCeleb repository - https://github.com/DASH-Lab/FakeAVCeleb/blob/main/models/MesoNet.py.
"""
import torch
import torch.nn as nn

from src import frontends

class MesoInception4(nn.Module):
    """
    Pytorch Implemention of MesoInception4
    Author: Honggu Liu
    Date: July 7, 2019
    """
    def __init__(self, num_classes=1, **kwargs):
        super().__init__()

        self.fc1_dim = kwargs.get("fc1_dim", 1024)
        input_channels = kwargs.get("input_channels", 3)
        self.num_classes = num_classes

        # Inception Layer 1
        self.Incption1_conv1 = nn.Conv2d(input_channels, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(input_channels, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(input_channels, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(input_channels, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)

        # Inception Layer 2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        # Normal Layers
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(self.fc1_dim, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def InceptionLayer1(self, input):
        x1 = self.Incption1_conv1(input)
        x2 = self.Incption1_conv2_1(input)
        x2 = self.Incption1_conv2_2(x2)
        x3 = self.Incption1_conv3_1(input)
        x3 = self.Incption1_conv3_2(x3)
        x4 = self.Incption1_conv4_1(input)
        x4 = self.Incption1_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)
        return y

    def InceptionLayer2(self, input):
        x1 = self.Incption2_conv1(input)
        x2 = self.Incption2_conv2_1(input)
        x2 = self.Incption2_conv2_2(x2)
        x3 = self.Incption2_conv3_1(input)
        x3 = self.Incption2_conv3_2(x3)
        x4 = self.Incption2_conv4_1(input)
        x4 = self.Incption2_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)
        return y

    def _compute_embedding(self, input):
        x = self.InceptionLayer1(input)
        x = self.InceptionLayer2(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling2(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.AdaptiveAvgPool1d(self.fc1_dim)(x)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def forward(self, input):
        x = self._compute_embedding(input)
        return x
    

class FrontendMesoInception4(MesoInception4):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs['device']
        frontend_name = kwargs.get("frontend_algorithm", [])
        self.frontend = frontends.get_frontend(frontend_name, self.device)

    def forward(self, x):
        x = self.frontend(x)
        return super().forward(x)

class FrontendMesoInception4_lfccmfcc(MesoInception4):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs['device']
        frontend_name = kwargs.get("frontend_algorithm", [])
        self.frontend_1 = frontends.get_frontend(frontend_name[0], self.device)
        self.frontend_2 = frontends.get_frontend(frontend_name[1], self.device)
        print(f"Using {frontend_name} frontend")

    def forward(self, x):
        x_1 = self.frontend_1(x)
        x_2 = self.frontend_2(x)
        x = torch.cat([x_1, x_2], 1)
        return super().forward(x)
    

class MesoContext(nn.Module):
    def __init__(self, num_classes=1, context="no", **kwargs):
        super().__init__()
        self.context = context
        self.fc1_dim = kwargs.get("fc1_dim", 1024)
        input_channels = kwargs.get("input_channels", 3)
        self.num_classes = num_classes

        self.embedding_dim = kwargs.get("embedding_dim", 100)
        self.context_fc_dim = 6

        # Inception Layer 1
        self.Incption1_conv1 = nn.Conv2d(input_channels, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(input_channels, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(input_channels, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(input_channels, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)

        # Inception Layer 2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        # Normal Layers
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.fc1_dim, 16)
        self.fc2 = nn.Linear(16, num_classes)

        self.hidden_size = 512
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)

        self.encoder_c = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),  # 3081 1540 777
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            # nn.Linear(512, 128),
            # # nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Linear(128, self.context_fc_dim),
            nn.LeakyReLU(0.1),
        )

        self.maxpool_c = nn.MaxPool1d(3, stride=2)

        self.fc2_o_c = nn.Sequential(
            nn.Linear(16 + self.context_fc_dim, 16),  # 16 + 6
            nn.LeakyReLU(0.1),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.1),
            # nn.Linear(512, 128),
            # # nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Linear(16, num_classes),
            # nn.LeakyReLU(0.1),
        )

    def InceptionLayer1(self, input):
        x1 = self.Incption1_conv1(input)
        x2 = self.Incption1_conv2_1(input)
        x2 = self.Incption1_conv2_2(x2)
        x3 = self.Incption1_conv3_1(input)
        x3 = self.Incption1_conv3_2(x3)
        x4 = self.Incption1_conv4_1(input)
        x4 = self.Incption1_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)
        return y

    def InceptionLayer2(self, input):
        x1 = self.Incption2_conv1(input)
        x2 = self.Incption2_conv2_1(input)
        x2 = self.Incption2_conv2_2(x2)
        x3 = self.Incption2_conv3_1(input)
        x3 = self.Incption2_conv3_2(x3)
        x4 = self.Incption2_conv4_1(input)
        x4 = self.Incption2_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)
        return y

    def _compute_embedding(self, input, c_e):
        if self.context == "no":
            x = self.InceptionLayer1(input)  # (Batch, 11, 128, 128)
            x = self.InceptionLayer2(x)  # (Batch, 12, 64, 64)

            x = self.conv1(x)  # (Batch, 16, 64 ,64)
            x = self.relu(x)
            x = self.bn1(x)
            x = self.maxpooling1(x)  # (Batch, 16, 32, 32)

            x = self.conv2(x)  # (Batch, 16, 32, 32)
            x = self.relu(x)
            x = self.bn1(x)
            x = self.maxpooling2(x)  # (Batch, 16, 8, 8)

            x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
            x = self.dropout(x)

            x = nn.AdaptiveAvgPool1d(self.fc1_dim)(x)

            x = self.fc1(x)  # (Batch, 16)  ### <-- o tu
            x_a = self.leakyrelu(x)
            x = self.dropout(x_a)
            x = self.fc2(x)

            return x

        elif self.context == "ct":
            x = self.InceptionLayer1(input)  # (Batch, 11, 128, 128)
            x = self.InceptionLayer2(x)  # (Batch, 12, 64, 64)

            x = self.conv1(x)  # (Batch, 16, 64 ,64)
            x = self.relu(x)
            x = self.bn1(x)
            x = self.maxpooling1(x)  # (Batch, 16, 32, 32)

            x = self.conv2(x)  # (Batch, 16, 32, 32)
            x = self.relu(x)
            x = self.bn1(x)
            x = self.maxpooling2(x)  # (Batch, 16, 8, 8)

            x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
            x = self.dropout(x)

            x = nn.AdaptiveAvgPool1d(self.fc1_dim)(x)

            x = self.fc1(x)  # (Batch, 16)  ### <-- o tu
            x_a = self.leakyrelu(x)
            x_c = self.encoder_c(c_e)
            x = torch.cat((x_a, x_c), 1)

            x = self.dropout(x)
            x = self.fc2_o_c(x)
            return x
        
    def forward(self, input, c_e):
        x = self._compute_embedding(input, c_e)
        return x


class FrontendMesoContext(MesoContext):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs['device']
        frontend_name = kwargs.get("frontend_algorithm", [])
        self.frontend = frontends.get_frontend(frontend_name, self.device)

    def forward(self, x, **kwargs):
        x = self.frontend(x)
        return super().forward(x, **kwargs)

class FrontendMesoContext_lfccmfcc(MesoContext):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs['device']
        frontend_name = kwargs.get("frontend_algorithm", [])
        self.frontend_1 = frontends.get_frontend(frontend_name[0], self.device)
        self.frontend_2 = frontends.get_frontend(frontend_name[1], self.device)
        print(f"Using {frontend_name} frontend")

    def forward(self, x, **kwargs):
        x_1 = self.frontend_1(x)
        x_2 = self.frontend_2(x)
        x = torch.cat([x_1, x_2], 1)
        return super().forward(x, **kwargs)
