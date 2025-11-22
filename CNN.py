import torch
import torch.nn as nn
import torch.nn.functional as F

# 构建步骤：
# 导入PyTorch相关模块
# 定义CNN模型类，继承nn.Module
# 在__init__中定义网络层
# 实现forward方法定义前向传播
# 创建模型实例
# 定义损失函数和优化器
# 训练模型

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层 (输入通道1, 输出通道32, 卷积核3x3)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # 卷积层 (输入通道32, 输出通道64, 卷积核3x3)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层 (64*7*7 -> 128)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 输出层 (128 -> 10)
        self.fc2 = nn.Linear(128, 10)
        # Dropout层
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # 添加层并应用ReLU激活函数
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征
        x = x.view(-1, 64 * 7 * 7)
        # 添加dropout
        x = self.dropout(x)
        # 添加全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    @classmethod
    def base_train(cls):
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, 10)  # 假设输入为224x224
        )
        return model
    
    @classmethod
    def train_model(cls):
        # 创建模型
        model = CNN()
        print(model)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 假设我们有数据加载器 train_loader 和 test_loader
        # 训练循环示例
        for epoch in range(10):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                # 清零梯度
                optimizer.zero_grad()
                # 前向传播
                outputs = model(images)
                # 计算损失
                loss = criterion(outputs, labels)
                # 反向传播
                loss.backward()
                # 更新权重
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

class CNNBuilder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        for (in_ch, out_ch, kernel) in config:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

    @classmethod
    def test(cls):
        config = [(3,64,3), (64,128,3), (128,256,5)]
        model = CNNBuilder(config)
        print(model)