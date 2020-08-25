from .Trainer import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gate(x, y):
    return 0.5

def shift(new, pre):
    return (new - pre)

class StreamTrainer(Trainer):

    def update(self):

        res = 0.0
        for i in range(self.data_loader.get_nstream()):
            data = self.data_loader.sampling_stream()

            self.optimizer.zero_grad()
            loss = self.model({
                    'batch_h': self.to_var(data['batch_h'], self.use_gpu),
                    'batch_t': self.to_var(data['batch_t'], self.use_gpu),
                    'batch_r': self.to_var(data['batch_r'], self.use_gpu),
                    'batch_y': self.to_var(data['batch_y'], self.use_gpu),
                    'mode': data['mode']
            })
            loss.backward()
            self.optimizer.step()
            res += loss.item()

        print(res)
