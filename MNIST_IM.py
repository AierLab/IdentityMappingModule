import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# 加载训练集和测试集
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=0)


class ZeroConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ZeroConvBatchNorm, self).__init__()

        # Define a zero convolutional layer with batch normalization
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv.weight.data.fill_(0)  # Initialize weights with zeros
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)  # Apply the zero convolution operation
        out = self.bn(out)   # Apply batch normalization
        return out

class IdentityMappingModule(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(IdentityMappingModule, self).__init__()

        layers = []
        for _ in range(6):  # Create 6 pairs of zero conv and batch norm layers
            zero_conv_bn = ZeroConvBatchNorm(hidden_channels, hidden_channels)
            layers.append(zero_conv_bn)
        self.identity_module = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        out = self.identity_module(x)  # Apply the sequence of zero conv and batch norm layers
        return out + x  # Implement skip connection by adding input tensor to output tensor
    
# 定义结合了IdentityMappingModule的神经网络模型
class SimpleNetWithIdentityModule(nn.Module):
    def __init__(self):
        super(SimpleNetWithIdentityModule, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.identity_module = IdentityMappingModule(64, 64)  # 使用IdentityMappingModule
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.identity_module(x)  # 使用IdentityMappingModule
        x = x.view(x.size(0), -1)  # 将 x 展平为 [batch_size, 64]
        x = torch.relu(x)
        x = self.fc3(x)
        return x


# 初始化网络、损失函数和优化器
# 初始化网络、损失函数和优化器
net = SimpleNetWithIdentityModule()

# 加载之前训练过的全连接层的权重，不改变其权重
PATH = "IdentityMappingModule/model_weights.pth"
pretrained_dict = torch.load(PATH)
model_dict = net.state_dict()
# 过滤出只需要的权重
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新当前模型的权重
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

# 冻结全连接层的权重，使其不参与训练
for param in net.fc1.parameters():
    param.requires_grad = False
for param in net.fc2.parameters():
    param.requires_grad = False
for param in net.fc3.parameters():
    param.requires_grad = False

# 定义只优化恒等映射层的优化器
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.identity_module.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.identity_module.parameters(), lr=0.001)  # 使用Adam优化器

# net = SimpleNetWithIdentityModule()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 训练网络
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')


# # 保存训练后的权重

# torch.save(net.state_dict(), PATH)
# print("Model weights saved to", PATH)


# # 加载模型权重并进行推断
# net = SimpleNetWithIdentityModule()
# net.load_state_dict(torch.load(PATH), strict = False)
# net.eval()


# 测试网络
correct = 0
total = 0
with torch.no_grad():   
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')