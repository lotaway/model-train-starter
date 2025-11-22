import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from utils import get_device

class TextDataset(Dataset):
    def __init__(self, texts, vocab, max_length):
        self.texts = texts
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 将文本转换为token索引
        tokens = [self.vocab.get(char, self.vocab['<unk>']) for char in text]
        # 填充或截断到固定长度
        if len(tokens) < self.max_length:
            tokens.extend([self.vocab['<pad>']] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        return torch.tensor(tokens)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # 准备输入和目标
        src = data[:-1]
        tgt = data[1:]
        
        # 生成mask
        tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(device)
        
        # 前向传播
        output = model(src, tgt, tgt_mask=tgt_mask)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

def test_multihead():
    # 使用 PyTorch 内置的多头注意力
    mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)

    # 假设输入形状是: (seq_len, batch_size, embed_dim)
    query = torch.randn(10, 32, 512)  # seq_len=10, batch_size=32, embed_dim=512
    key = torch.randn(10, 32, 512)
    value = torch.randn(10, 32, 512)

    # 前向传播
    attn_output, attn_weights = mha(query, key, value)

def test():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 超参数
    batch_size = 32
    max_length = 100
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    epochs = 10
    learning_rate = 0.0001
    
    # 设备设置
    device = get_device()

    # 示例数据
    texts = [
        "Hello, how are you?",
        "I am fine, thank you.",
        "What is your name?",
        "My name is AI.",
        "Nice to meet you."
    ]
    
    # 构建词汇表
    vocab = {'<pad>': 0, '<unk>': 1}
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    
    # 创建数据集和数据加载器
    dataset = TextDataset(texts, vocab, max_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = TransformerModel(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    for epoch in range(1, epochs + 1):
        avg_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.6f}')
    
    # 保存模型
    torch.save(model.state_dict(), 'transformer_model.pth')
    
    # 示例推理
    model.eval()
    with torch.no_grad():
        test_text = "Hello"
        test_tokens = [vocab.get(char, vocab['<unk>']) for char in test_text]
        test_tokens = torch.tensor(test_tokens).unsqueeze(1).to(device)
        
        # 生成预测
        output = model(test_tokens, test_tokens)
        predicted = output.argmax(dim=-1)
        
        # 将预测结果转换回文本
        idx_to_char = {v: k for k, v in vocab.items()}
        predicted_text = ''.join([idx_to_char[idx.item()] for idx in predicted.squeeze()])
        print(f"Input: {test_text}")
        print(f"Predicted: {predicted_text}")