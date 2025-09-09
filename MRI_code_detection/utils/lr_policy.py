import torch

class WarmUpPolyLR(object):
    
    def __init__(self, optimizer, start_lr, lr_power, total_iters, warmup_steps):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters
        self.warmup_steps = warmup_steps
        self.iter = 0
    
    def get_lr(self, iter):
        if iter < self.warmup_steps:
            return self.start_lr * (iter / self.warmup_steps)
        else:
            return self.start_lr * ((1 - iter / self.total_iters) ** self.lr_power)
    
    def step(self):
        self.iter += 1
        lr = self.get_lr(self.iter)
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = lr
