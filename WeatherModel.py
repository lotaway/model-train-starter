import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset, config
import numpy as np
import matplotlib.pyplot as plt
from utils import DataLoaderWrapper, fit
# print(config.HF_DATASETS_CACHE)

class WeatherModel(nn.Module):
    def __init__(self, input_features, labels):
        super().__init__()

        self.x = torch.tensor(input_features, dtype=float)
        self.y = torch.tensor(labels, dtype=float)

    @classmethod
    def train_data(cls):
        dataset = load_dataset("kanishka089/weather")
        # dataset = load_dataset("csv", data_files={"weather_classification_data.csv"})
        # df = pd.read_csv(dataset)
        df = dataset["train"].to_pandas()
        return df

    @classmethod
    def train_model_manually(cls):
        df = cls.train_data()
        input_features = df.drop("actual", axis=1).values
        labels = df["actual"].values

        model = WeatherModel(input_features, labels)

        input_size = input_features.shape[1]
        hidden_size = 128
        output_size = 1
        learning_rate = 0.001

        weights = torch.randn((input_size, hidden_size), dtype=float, requires_grad=True)
        biases = torch.randn((1, hidden_size), dtype=float, requires_grad=True)

        weights2 = torch.randn((hidden_size, 1), dtype=float, requires_grad=True)
        biases2 = torch.randn((output_size, 1), dtype=float, requires_grad=True)
        losses = []

        for i in range(1000):
            # 计算隐层，加入激活函数
            hidden = model.x.mm(weights) + biases
            hidden = torch.relu(hidden)
            # 预测结果
            predictions = hidden.mm(weights2) + biases2
            # 计算损失
            loss = torch.mean((predictions - model.y) ** 2)
            losses.append(loss.data.numpy())

            if i % 100 == 0:
                print(f"Epoch {i}: Loss {loss.data.numpy()}")

            # 更新参数
            weights.data.add_(-learning_rate * weights.grad.data)
            biases.data.add_(-learning_rate * biases.grad.data)
            weights2.data.add_(-learning_rate * weights2.grad.data)
            biases2.data.add_(-learning_rate * biases2.grad.data)

            # 清空梯度
            weights.grad.data.zero_()
            biases.grad.data.zero_()
            weights2.grad.data.zero_()
            biases2.grad.data.zero_()

    @classmethod
    def train_model_simple(cls):
        df = cls.train_data()
        print("Original data shape:", df.shape)
        print("Original columns:", df.columns)
        
        # Convert date column to numeric features if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df = df.drop('date', axis=1)
        
        # Ensure all columns are numeric and handle NaN values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Remove rows with NaN values
            df = df.dropna(subset=[col])
        
        print("Data shape after cleaning:", df.shape)
        print("Data types:", df.dtypes)
        
        # Normalize the data
        df = (df - df.mean()) / df.std()
        
        # Check for any remaining NaN values
        if df.isnull().any().any():
            print("Warning: NaN values still present after normalization")
            print(df.isnull().sum())
            return
        
        input_features = df.drop("Temperature", axis=1).values.astype(np.float32)
        labels = df["Temperature"].values.astype(np.float32)

        print("Input features shape:", input_features.shape)
        print("Labels shape:", labels.shape)
        
        input_size = input_features.shape[1]
        hidden_size = 64  # Reduced from 128
        output_size = 1
        batch_size = 32  # Increased from 16
        learning_rate = 0.0001  # Reduced from 0.001

        my_nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(my_nn.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        losses = []
        for i in range(1000):
            batch_loss = []
            my_nn.train()  # Set model to training mode
            
            for start in range(0, len(input_features), batch_size):
                end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
                xx = torch.tensor(input_features[start:end], dtype=torch.float32)
                yy = torch.tensor(labels[start:end], dtype=torch.float32)
                
                optimizer.zero_grad()
                prediction = my_nn(xx)
                loss = criterion(prediction.squeeze(), yy)
                
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at epoch {i}, batch {start}")
                    print("Input features stats:", xx.mean().item(), xx.std().item())
                    print("Labels stats:", yy.mean().item(), yy.std().item())
                    break
                
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(my_nn.parameters(), max_norm=1.0)
                optimizer.step()
                batch_loss.append(loss.item())

            if batch_loss:  # Only proceed if we have valid losses
                mean_batch_loss = np.mean(batch_loss)
                losses.append(mean_batch_loss)
                scheduler.step(mean_batch_loss)
                
                if i % 10 == 0:  # Print more frequently
                    print(f"Epoch {i}: Loss {mean_batch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        x = torch.tensor(input_features, dtype=torch.float)
        predict = my_nn(x).data.numpy()
        
        # 准备绘图数据

        # 提取日期信息
        dates = pd.to_datetime(df['date'])
        years = dates.dt.year
        months = dates.dt.month
        days = dates.dt.day
        date_strings = [f"{int(year)}-{int(month):02d}-{int(day):02d}" 
                       for year, month, day in zip(years, months, days)]
        
        cls.plot_results({
            'model': my_nn,
            'predictions': predict,
            'actual': labels,
            'dates': date_strings,
            'losses': losses
        })

    @classmethod
    def plot_results(cls, results):
        plt.figure(figsize=(12, 6))
        plt.plot(results['dates'], results['actual'], label='Actual', color='blue')
        plt.plot(results['dates'], results['predictions'], label='Predicted', color='red')
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.title('Weather Prediction Results')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

class Mnist_NN(nn.Module):

    @classmethod
    def get_data(cls):
        x_train = np.array([[1, 2], [3, 4]])  # 示例数据
        y_train = np.array([1, 2])
        x_valid = np.array([[5, 6], [7, 8]])
        y_valid = np.array([3, 4])

        x_train, y_train, x_valid, y_valid = map(
            torch.tensor, (x_train, y_train, x_valid, y_valid)
        )
        n, c = x_train.shape
        print("Training data shape:", x_train.shape)
        return x_train, y_train, x_valid, y_valid, n, c

    @classmethod
    def train_model(cls):
        x_train, y_train, x_valid, y_valid, n, c = cls.get_data()
        train_dl, valid_dl = DataLoaderWrapper(x_train, y_train, x_valid, y_valid, bs=32)

        loss_func = F.cross_entropy

        bs = 64
        # xb = x_train[0:bs]
        # yb = y_train[0:bs]
        # weights = torch.randn([784, 10], dtype=torch.float, requires_grad=True)
        # bias = torch.zeros(10, requires_grad=True)

        # result = xb.mm(weights) + bias
        # print(result)
        model = Mnist_NN()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        fit(25, model, loss_func, optimizer, train_dl, valid_dl)
        print(model, optimizer)

    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x