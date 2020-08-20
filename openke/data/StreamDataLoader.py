# coding:utf-8
import os
import ctypes
import numpy as np

import random
from Graph import Graph

class TrainDataSampler(object):

    def __init__(self, nbatches, datasampler):
        self.nbatches = nbatches
        self.datasampler = datasampler
        self.batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.batch += 1 
        if self.batch > self.nbatches:
            raise StopIteration()
        return self.datasampler()

    def __len__(self):
        return self.nbatches

class StreamDataLoader(object):

    def __init__(self, 
        in_path = "./",
        tri_file = None,
        ent_file = None,
        rel_file = None,
        stream_file = None,
        batch_size = None,
        stream_size = None,
        nbatches = None,
        nstreams = None,
        threads = 8,
        sampling_mode = "normal",
        bern_flag = False,
        filter_flag = True,
        neg_ent = 1,
        neg_rel = 0):
        
        self.in_path = in_path
        self.tri_file = tri_file
        self.ent_file = ent_file
        self.rel_file = rel_file
        self.stream_file = stream_file
        if in_path != None:
            self.tri_file = in_path + "train2id.txt"
            self.ent_file = in_path + "entity2id.txt"
            self.rel_file = in_path + "relation2id.txt"
            self.stream_file = in_path + "stream2id.txt"
        """set essential parameters"""
        self.work_threads = threads
        self.nbatches = nbatches
        self.nstreams = nstreams
        self.batch_size = batch_size
        self.stream_size = stream_size
        self.bern = bern_flag
        self.filter = filter_flag
        self.negative_ent = neg_ent
        self.negative_rel = neg_rel
        self.sampling_mode = sampling_mode
        self.cross_sampling_flag = 0
        self.read()
        self.graph = Graph()
        self.batch_index = 0
        self.stream_index = 0
        self.tripleTotal = 0
        self.streamTripleTotal = 0
        self.trainList = 0
        self.streamList = 0

    def read(self):
        with open(self.rel_file, "r") as f:
            line = f.readline()
            self.relTotal = int(line)

        with open(self.ent_file, "r") as f:
            line = f.readline()
            self.entTotal = int(line)

        with open(self.tri_file, "r") as f:
            line = f.readline()
            self.tripleTotal = int(line)
            self.trainList = np.zeros((self.tripleTotal, 3), dtype=int)
            index = 0
            while True:
                line = f.readline()
                if line:
                    items = line.split()
                    self.trainList[index][0] = int(items[0])
                    self.trainList[index][1] = int(items[1])
                    self.trainList[index][2] = int(items[2])
                    index += 1
                else:
                    break

        with open(self.stream_file, "r") as f:
            line = f.readline()
            self.streamTripleTotal = int(line)
            self.streamList = np.zeros((self.streamTripleTotal, 3), dtype=int)
            index = 0
            while True:
                line = f.readline()
                if line:
                    items = line.split()
                    self.streamList[index][0] = int(items[0])
                    self.streamList[index][1] = int(items[1])
                    self.streamList[index][2] = int(items[2])
                    index += 1
                else:
                    break
        
        self.graph.init(self.streamTripleTotal, self.streamList)

        if self.batch_size == None:
            self.batch_size = self.tripleTotal // self.nbatches
        if self.nbatches == None:
            self.nbatches = self.tripleTotal // self.batch_size

        if self.stream_size == None:
            self.stream_size = self.streamTripleTotal // self.nstreams
        if self.nstreams == None:
            self.nstreams = self.streamTripleTotal // self.stream_size

        self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
        self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)

        self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)

        self.stream_seq_size = self.stream_size * (1 + self.negative_ent + self.negative_rel)
        self.stream_h = np.zeros(self.stream_seq_size, dtype=np.int64)
        self.stream_t = np.zeros(self.stream_seq_size, dtype=np.int64)
        self.stream_r = np.zeros(self.stream_seq_size, dtype=np.int64)
        self.stream_y = np.zeros(self.stream_seq_size, dtype=np.float32)


    def corrupt(self, data):
        # data's shape :(batchsize,)
        return [random.randint(0, self.entTotal-1) for i in range(self.batch_size)]

    def sampling(self):
        if self.batch_index == self.nbatches:
            self.batch_index = 0

        index = self.batch_index * self.batch_size 
        self.batch_index += 1

        # positive sampling
        for i in range(self.batch_size):
            self.batch_h[i] = self.trainList[index+i,0]
            self.batch_t[i] = self.trainList[index+i,1]
            self.batch_r[i] = self.trainList[index+i,2]

        # negative sampling
        for i in range(1, self.negative_ent + 1):
            neg_index = self.batch_size * i
            rand = random.random()
            if rand > 0.5:
                # corrupt tail
                cor_tail = self.corrupt(self.trainList[index+i,1])
                for ii, neg_data in enumerate(cor_tail):
                    self.batch_t[neg_index+ii] = neg_data
                self.batch_h[neg_index:neg_index+self.batch_size] = self.trainList[index+i,0]
                self.batch_r[neg_index:neg_index+self.batch_size] = self.trainList[index+i,2]
            else :
                # corrupt head
                cor_head = self.corrupt(self.trainList[index+i,0])
                for ii, neg_data in enumerate(cor_head):
                    self.batch_h[neg_index+ii] = neg_data
                self.batch_t[neg_index:neg_index+self.batch_size] = self.trainList[index+i,1]
                self.batch_r[neg_index:neg_index+self.batch_size] = self.trainList[index+i,2]

        return {
            "batch_h": self.batch_h,
            "batch_t": self.batch_t,
            "batch_r": self.batch_r,
            "batch_y": self.batch_y,
            "mode": "normal"
        }


    # Todo : stream architecture
    def sampling_stream(self):
        if self.stream_index == self.nstreams:
            self.stream_index = 0

        index = self.stream_index * self.stream_size 
        self.stream_index += 1

        # positive sampling
        for i in range(self.stream_size):
            self.stream_h[i] = self.streamList[index+i,0]
            self.stream_t[i] = self.streamList[index+i,1]
            self.stream_r[i] = self.streamList[index+i,2]

        return {
            "batch_h": self.stream_h,
            "batch_t": self.stream_t,
            "batch_r": self.stream_r,
            "batch_y": self.stream_y,
            "mode": "normal"
        }


    """interfaces to set essential parameters"""

    def set_work_threads(self, work_threads):
        self.work_threads = work_threads

    def set_in_path(self, in_path):
        self.in_path = in_path

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.nbatches = self.tripleTotal // self.batch_size

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_bern_flag(self, bern):
        self.bern = bern

    def set_filter_flag(self, filter):
        self.filter = filter

    """interfaces to get essential parameters"""

    def get_batch_size(self):
        return self.batch_size

    def get_ent_tot(self):
        return self.entTotal

    def get_rel_tot(self):
        return self.relTotal

    def get_triple_tot(self):
        return self.tripleTotal

    def __iter__(self):
        if self.sampling_mode == "normal":
            return TrainDataSampler(self.nbatches, self.sampling)
        else:
            return TrainDataSampler(self.nbatches, self.cross_sampling)

    def __len__(self):
        return self.nbatches
