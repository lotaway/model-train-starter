from array import array

import matplotlib.pyplot
# from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import math
import torch
from numexpr.expressions import double

from LinearRegressionModel import LinearRegressionModel
from utils import tokenize, get_device
from WeatherModel import WeatherModel


def paint():
    # 获得0到2π之间的ndarray对象
    x = np.arange(0, math.pi * 2, 0.05)
    y = np.sin(x)
    plt.plot(x, y)
    plt.xlabel("angle")
    plt.ylabel("sine")
    plt.title('sine wave')
    plt.show()


def paint2():
    # Generate 100 random data points along 3 dimensions
    x, y, scale = np.random.randn(3, 100)
    fig, ax = plt.subplots()

    # Map each onto a scatterplot we'll create with Matplotlib
    ax.scatter(x=x, y=y, c=scale, s=np.abs(scale) * 500)
    ax.set(title="Some random data, created with JupyterLab!")
    plt.show()


def main1():
    device = get_device()
    tensor = torch.randn(2, 2).to(device)
    print(tensor)
    paint()
    paint2()
    tokens = tokenize("RagFlow对话系统特点与应用")
    print(tokens)
    init_torch(torch.randn(5, 3).numpy())

def main2():
    print("Start in main")
    x_train, y_train = LinearRegressionModel.train_data()
    print("Train data get done")
    model = LinearRegressionModel.train_model(x_train, y_train)
    print(model)

def main():
    print("Start in main")
    WeatherModel.train_model_simple()
    print("Train data get done")


def init_torch(_x: np.ndarray):
    x = torch.from_numpy(_x)
    if not x:
        x = torch.zeros(4, 3, dtype=double)
    print(x.size())
    y = x.view(12)
    y.requires_grad = True
    z = y.view(-1, 6)
    y.requires_grad = True
    z2 = y + z
    print("the z2 grad", z2.requires_grad)  # True
    return x, y, z, y + z


def test_backward():
    # torch.zeros(3, 2).is_leaf
    return (
            torch.randn(3, 4, requires_grad=True) +
            torch.randn(3, 4, requires_grad=True)
    ).sum().backward()


if __name__ == "__main__":
    main()
