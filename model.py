import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Cycle-DiSCUSS
Cycle- Dimensionality Supervised CT-US Synchronize System

"""


# 定义一个基本的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入和输出的通道数不同，或者步长不为1，则需要进行下采样
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

# 定义ResNet架构
class ResNet2D(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet2D, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 32)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 实例化ResNet模型
def ResNet18():
    return ResNet2D(ResidualBlock, [2, 2, 2, 2])

# 定义一个基本的3D残差块
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 如果输入和输出的通道数不同，或者步长不为1，则需要进行下采样
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

# 定义3D ResNet架构
class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, 32)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 实例化3D ResNet模型
def ResNet18_3D():
    return ResNet3D(ResidualBlock3D, [2, 2, 2, 2])


# 融合2D和3D特征的模型
class FeatureFusion2D3DModel(nn.Module):
    def __init__(self, num_classes=6):
        super(FeatureFusion2D3DModel, self).__init__()
        self.resnet2d = ResNet18()
        self.resnet3d = ResNet18_3D()
        self.fc = nn.Linear(self.resnet2d.fc.out_features + self.resnet3d.fc.out_features, num_classes)

    def forward(self, x_2d, x_3d):
        # 提取2D特征
        features_2d = self.resnet2d(x_2d)
        # 提取3D特征
        features_3d = self.resnet3d(x_3d)

        # 拼接特征
        fused_features = torch.cat((features_2d, features_3d), dim=1)
        # 通过全连接层进行分类
        output = self.fc(fused_features)
        return output


# 融合3D和3D特征的模型
class FeatureFusion3D3DModel(nn.Module):
    def __init__(self, num_classes=12):
        super(FeatureFusion3D3DModel, self).__init__()
        self.resnet3d_us = ResNet18_3D()
        self.resnet3d_ct = ResNet18_3D()
        self.fc = nn.Linear(self.resnet3d_us.fc.out_features + self.resnet3d_ct.fc.out_features, num_classes)

    def forward(self, x_3d_us, x_3d_ct):
        # 提取2D特征
        features_3d_us = self.resnet3d_us(x_3d_us)
        # 提取3D特征
        features_3d_ct = self.resnet3d_ct(x_3d_ct)

        # 拼接特征
        fused_features = torch.cat((features_3d_us, features_3d_ct), dim=1)
        # 通过全连接层进行分类
        output = self.fc(fused_features)
        return output




# 示例用法
if __name__ == "__main__":
    # 创建一个随机输入张量
    input_tensor_2d = torch.randn(1, 3, 64,64)  # 假设输入是32x32的RGB图像
    input_tensor_3d = torch.randn(1, 3, 64,64, 64)  # 假设输入是16x32x32的RGB视频帧
    # 实例化融合模型
    model = FeatureFusion2D3DModel()
    # 前向传播
    output = model(input_tensor_2d, input_tensor_3d)
    print("Output shape:", output.shape)


    # 创建一个随机输入张量
    input_tensor_3d_us = torch.randn(1, 3, 64,64,64)  # 假设输入是32x32的RGB图像
    input_tensor_3d_ct = torch.randn(1, 3, 64,64, 64)  # 假设输入是16x32x32的RGB视频帧
    # 实例化融合模型
    model = FeatureFusion3D3DModel()
    # 前向传播
    output = model(input_tensor_3d_us, input_tensor_3d_ct)
    print("Output shape:", output.shape)