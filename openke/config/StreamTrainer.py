from .Trainer import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

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

            # Todo : implement Refresh and Percolate

            # Todo : update graph

        return res

    def run_stream(self, iteration=1):
        print("start train stream file")
        training_range = tqdm(range(iteration))
        for epoch in training_range:
                res = self.update()
                training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
