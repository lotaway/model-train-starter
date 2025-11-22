from typing import Optional

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn


def load_data():
    # 定义数据变换，可增加更多数据集
    transform = transforms.Compose([
        # for vision dataset
        # transforms.RandomRotation(45), # 随机旋转，范围为[-45, 45]
        # transforms.CenterCrop(224), # 中心裁剪
        # transforms.RandomHorizontalFlip(p=0.5), # 概率翻转
        # transforms.RandomVerticalFlip(p=0.5), # 概率翻转
        # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1), # 随机调整亮度、对比度、饱和度、色调
        # transforms.RandomGrayscale(p=0.1), # 随机灰度化
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载训练和测试数据集
    train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    return (train_loader, test_loader)


class SimpleNN(nn.Module):
    def __init__(self, input_dim: Optional[int], input_dim_2: Optional[int], out_dim: Optional[int], ):
        super(SimpleNN, self).__init__()
        _output_dim = input_dim_2.__or__(128)
        self.fc1 = nn.Linear(input_dim.__or__(28 * 28), _output_dim)
        self.fc2 = nn.Linear(_output_dim, out_dim.__or__(10))

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def start_train():
    # 实例化模型、定义优化器和损失函数
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # torch.nn.Module
    model = SimpleNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    (train_loader, test_loader) = load_data()

    # 开始训练
    for epoch in range(5):
        train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch [{epoch + 1}/5] completed.')
