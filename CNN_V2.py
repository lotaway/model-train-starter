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
        input_w = input_h = 28
        out_chainnels = 16
        self. Conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, # 1 * 28 * 28
                out_channels=out_chainnels, 
                kernel_size=5, 
                stride=1, 
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # new_x = (h - filter_size + 2 * padding) / stride + 1)
        # After Conv2d & ReLU & MaxPool2d -> 16 * 14 * 14
        out_chainnels_2 = 32
        self.Conv2 = nn.Sequential(
            nn.Conv2d(out_chainnels, out_chainnels_2, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) # -> 32 * 7 * 7
        self.fc1 = nn.Linear(input_w/2/2*input_h/2/2*out_chainnels_2, 10)
    
    # need batch_size in x, such as x = torch.randn(batch_size=32, 1, 28, 28)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = x.view(x.size(0), -1) # 展平给全连接层
        x = self.fc1(x)
        return x
    
    @classmethod
    def train_model(cls, num_epochs, train_loader, test_loader):
        model = CNN()
        criterion = nn.CrossEntropyLoss()
        opimazier = torch.optim.Adam(model.parameters(), lr=0.001)
        batch_size = 32
        
        for epoch in range(num_epochs):
            train_rights = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                model.train()
                output = model(data)
                loss = criterion(output, target)
                opimazier.zero_grad()
                loss.backward()
                opimazier.step()
                right = CNN.accuracy(output, target)
                train_rights.append(right)
                
                if batch_idx % 100 == 0:
                    model.eval()
                    val_rights = []
                    for (data, target) in test_loader:
                        output = model(data)
                        right = CNN.accuracy(output, target)
                        val_rights.append(right)
                        
                    train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                    val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
                    print('Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tTrain Acc: {:.2f}%\tVal Acc: {:.2f}%'.format(
                        epoch, 
                        batch_idx * batch_size,
                        len(train_loader.dataset),
                        100. * batch_idx * len(train_loader),
                        loss.data,
                        100. * train_r[0].numpy() / train_r[1],
                        100. * val_r[0].numpy() / val_r[1],
                    ))
            
        print(model)
    
    @classmethod
    def accuracy(predictions, labels):
        pred = torch.max(predictions.data, 1)[1]
        rights = pred.eq(labels.data.view_as(pred)).sum()
        return rights, len(labels)