from nltk import pos_tag, download
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import DataLoader, TensorDataset
import platform
import importlib
import numpy as np

# if not download('punkt_tab'):
#     print("Download punkt_tab not done")

def test():
    text = "NLP is a fascinating field of study."
    return tokenize(text)

def tokenize(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    print(pos_tags)
    return tokens

def get_device():
    device = None
    if platform.system() == 'Windows':
        torch_directml = importlib.import_module("torch_directml")
        device = torch_directml.device()
    if device:
        print("Using GPU ms ml")
        return device
    # check if GPU device work
    match True:
        case torch.backends.mps.is_available():  # apple silicon
            print("Using GPU mps")
            return torch.device("mps")  # use AMD Metal Performance Shaders ?
        case torch.cuda.is_available():  # nvidia
            print("Using GPU nvidia")
            return torch.device("cuda:0")
        case torch.hip.is_available():  # amd
            print("Using GPU hip")
            return torch.device("hip")
        case _:
            print("Using CPU")
            return torch.device("cpu")

def DataLoaderWrapper(x_train, y_train, x_valid, y_valid, bs):
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    return train_dl, train_dl, valid_dl, valid_dl

def DataLoaderWrapper(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def loss_batch(model, loss_fn, xb, yb, opt=None):
    loss = loss_fn(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(steps, model, loss_fn, optimizer, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_fn, xb, yb, optimizer)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_fn, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(f"当前Step {str(step)}, validation loss: {val_loss:.4f}")