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
        stream1 = self.data_loader.sampling_stream()
        
        batch_h = stream1['batch_h']
        batch_r = stream1['batch_r']
        batch_t = stream1['batch_t']
        stream = list()
        for i in range(len(batch_h)):
            stream.append([batch_h[i], batch_r[i], batch_t[i]])
        stream = np.asarray(stream)

        #for data in self.data_loader.getStream():
        loss = self.model({
                'batch_h': self.to_var(stream[:,0], self.use_gpu),
                'batch_t': self.to_var(stream[:,2], self.use_gpu),
                'batch_r': self.to_var(stream[:,1], self.use_gpu),
                'batch_y': self.to_var(stream1['batch_y'], self.use_gpu),
                'mode': stream1['mode'],
        })
        loss.sum().backward()

        self._model = self.model.model
        prev_emb_h = self._model.ent_embeddings # [stream[:,0]]
        prev_emb_t = self._model.ent_embeddings # [stream[:,2]]
        prev_emb_r = self._model.rel_embeddings # [stream[:,1]]

        self.optimizer.step()

        N1_head = self.data_loader.graph.head_dic
        N1_tail = self.data_loader.graph.tail_dic

        for (h, r, t) in stream:
            print(h, r, t, self._model.ent_embeddings)
            # update head embedding
            self._model.ent_embeddings[[h]] = 1
            print(self._model.ent_embeddings)
            #self._model.ent_embeddings[h] = gate(h, r) * (self._model.ent_embeddings[t] - self._model.rel_embeddings[r] - shift(self._model.ent_embeddings[t], prev_emb_t[t])) + (1 - gate(h, r)) * prev_emb_h[h]
            for n1 in N1_head[h]:
                self._model.ent_embeddings[n1[2]] = gate(h, n1[1]) * (self._model.ent_embeddings[h] - self._model.rel_embeddings[n1[1]] - shift(self._model.ent_embeddings[h], prev_emb_h[h])) + (1 - gate(h, n1[1])) * prev_emb_h[n1[2]]

            # update trai embedding
            self._model.ent_embeddings[t] = gate(t, r) * (self._model.ent_embeddings[h] + self._model.rel_embeddings[r] + shift(self._model.ent_embeddings[h], prev_emb_h[h])) + (1 - gate(t, r)) * prev_emb_t[t]
            for n1 in N1_tail[t]:
                self._model.ent_embeddings[n1[0]] = gate(t, n1[1]) * (self._model.ent_embeddings[t] + self._model.rel_embeddings[n1[1]] + shift(self._model.ent_embeddings[t], prev_emb_t[t])) + (1 - gate(t, n1[1])) * prev_emb_h[n1[0]]

            # update relation embedding
            self._model.rel_embeddings[r] = gate(r, r) * (self._model.ent_embeddings[t] - self._model.ent_embeddings[h]) + (1 - gate(r, r)) * prev_emb_r[t]    

