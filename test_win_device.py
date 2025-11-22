import torch_directml
import torch
import time

dml = torch_directml.device()
print("dml: " + format(dml))


class MyModule():
    size: int

    def __init__(self, _size=8192 * 2):
        self.size = _size

    def bench(self, device):
        print(f"Running on {device}")
        a = torch.rand((self.size, self.size), device=device)
        b = torch.rand((self.size, self.size), device=device)
        start = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        print(f"{device} matrix multiplication:", time.time() - start, "Seconds")


def test_perform():
    module= MyModule()
    # CPU
    module.bench("cpu")
    # DirectML
    module.bench(dml)


test_perform()
