import torch
from torch import nn
import torch.nn.functional as F


# class BasicBlock(nn.Module):
#     def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
#         super(BasicBlock, self).__init__()
#         # 残差部分
#         self.layer = nn.Sequential(
#             nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True), # 原地替换 节省内存开销
#             nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
#             nn.BatchNorm2d(out_channels)
#         )
#         # shortcut 部分
#         # 由于存在维度不一致的情况 所以分情况
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 # 卷积核为1 进行升降维
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
# #         print('shape of x: {}'.format(x.shape))
#         out = self.layer(x)
# #         print('shape of out: {}'.format(out.shape))
# #         print('After shortcut shape of x: {}'.format(self.shortcut(x).shape))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# # 采用bn的网络中，卷积层的输出并不加偏置
# class ResNet18(nn.Module):
#     def __init__(self, BasicBlock, num_classes=10) -> None:
#         super(ResNet18, self).__init__()
#         self.in_channels = 64
#         # 第一层作为单独的 因为没有残差快
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         # conv2_x
#         self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])
#         # self.conv2_2 = self._make_layer(BasicBlock,64,[1,1])

#         # conv3_x
#         self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])
#         # self.conv3_2 = self._make_layer(BasicBlock,128,[1,1])

#         # conv4_x
#         self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])
#         # self.conv4_2 = self._make_layer(BasicBlock,256,[1,1])

#         # conv5_x
#         self.conv5 = self._make_layer(BasicBlock,512,[[2,1],[1,1]])
#         # self.conv5_2 = self._make_layer(BasicBlock,512,[1,1])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)

#     #这个函数主要是用来，重复同一个残差块
#     def _make_layer(self, block, out_channels, strides):
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels
#         return nn.Sequential(*layers)
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.conv5(out)

# #       out = F.avg_pool2d(out,7)
#         out = self.avgpool(out)
#         out = out.reshape(x.shape[0], -1)
#         # out = self.fc(out)
#         return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核 6个
        self.conv1 = nn.Conv2d(1, 3, (3, 1))
        self.conv2 = nn.Conv2d(3, 32, (2, 1))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(608, 120)
        self.fc2 = nn.Linear(120, 80)
        # Vself.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling,先卷积+relu，再2 x 2下采样
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 1))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 1))
        #将特征向量扁平成行向量
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    net = Net()
    a = torch.randn((800, 1, 80, 1))
    b = net(a)
    print(b.shape)