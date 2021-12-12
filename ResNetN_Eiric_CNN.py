import torch
import numpy
import torch.nn as nn

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_data = torchvision.datasets.ImageFolder(root='./data/Eiric/TrainType4/train_data4', transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Resize((32, 32))
])

test_data = torchvision.datasets.ImageFolder(root='./data/Eiric/test_data', transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True)

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.shape)

# ResNet에 필요한 연산 따로 정의하지 않아 라이브러리에서 불러옴
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x) # 3x3 stride = 2
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out) # 3x3 stride = 1
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes) #conv1x1(64,64)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)#conv3x3(64,64)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion) #conv1x1(64,256)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x) # 1x1 stride = 1
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out) # 3x3 stride = stride
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out) # 1x1 stride = 1
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x.shape =[1, 64, 128,128]
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        # x.shape =[1, 128, 32,32]
        x = self.layer2(x)
        # x.shape =[1, 256, 32,32]
        x = self.layer3(x)
        # x.shape =[1, 512, 16,16]
        x = self.layer4(x)
        # x.shape =[1, 1024, 8,8]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# # resnet 50
# resnetN = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 721, True).to(device) 

# resnet 152
resnetN = ResNet(resnet.Bottleneck, [3, 8, 36, 3], 721, True).to(device) 

# resnetN = resnet50(num_classes=4)
# resnetN.to("cuda:0")
# print(resnetN)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(resnetN.parameters())
# optimizer = torch.optim.SGD(resnetN.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

def acc_check(net, test_set, epoch, save=1):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = (100 * correct / total)
    print('Accuracy of the test images: %d %%' % acc)
    if save:
        torch.save(net.state_dict(), "./model/Eiric_TrainType4/ResNet50_Eiric_epoch_{}_acc_{}.pth".format(epoch, int(acc)))
    return acc

print(len(train_loader))
epochs = 120

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnetN(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 0:  # print every 5 mini-batches
            print('[%d, %5d, %d%%] loss: %.3f' %
                  (epoch + 1, i + 1, 100 * (i + 1) / len(train_loader), running_loss / 5))
            running_loss = 0.0
    lr_sche.step()
    if epoch % 30 == 0:
      acc = acc_check(resnetN, test_loader, epoch, save=1)
    else:
      acc = acc_check(resnetN, test_loader, epoch, save=0)

print('Finished Training')


# 생성된 모델의 정확도를 측정
# new_resnetN = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 4, True).to(device)
# new_resnetN.load_state_dict(torch.load('./model/Eiric_TrainType4/ResNet50_Eiric_epoch_120_acc_31.pth'))
# with torch.no_grad():
#   for num, data in enumerate(test_loader):
#     imgs, label = data
#     imgs = imgs.to(device)
#     label = label.to(device)

#     prediction = new_resnetN(imgs)
#     print("check", torch.argmax(prediction, 1), label)
#     correct_prediction = torch.argmax(prediction, 1) == label

#     accuracy = correct_prediction.float().mean()
#     print('Accuracy:', accuracy.item())