import torch

class DataParallelTask:
    def __init__(self):
        # 初始化分布式训练
        torch.distributed.init_process_group(backend='nccl')
        # 设置 GPU 设备
        device = torch.device(f'cuda:{torch.distributed.get_rank()}')
        # torch.distributions.distribution.Distribution()
        # 构造模型并分布式
        model = torch.nn.Linear(10, 5).to(device)
        self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    def train(self):
        # TODO 训练代码
        print(self.model)