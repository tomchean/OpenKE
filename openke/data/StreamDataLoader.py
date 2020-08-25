# coding:utf-8
import os
import ctypes
import numpy as np

import random
from .Graph import Graph

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
        
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """argtypes"""
        self.lib.sampling.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64
        ]

        self.lib.sampling_stream.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64
        ]

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


        self.graph = Graph()
        self.batch_index = 0
        self.stream_index = 0
        self.tripleTotal = 0
        self.streamTripleTotal = 0
        self.trainList = 0
        self.streamList = 0
        
        self.read()

    def read(self):
        if self.in_path != None:
            self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
        else:
            self.lib.setTrainPath(ctypes.create_string_buffer(self.tri_file.encode(), len(self.tri_file) * 2))
            self.lib.setEntPath(ctypes.create_string_buffer(self.ent_file.encode(), len(self.ent_file) * 2))
            self.lib.setRelPath(ctypes.create_string_buffer(self.rel_file.encode(), len(self.rel_file) * 2))
        
        self.lib.setBern(self.bern)
        self.lib.setWorkThreads(self.work_threads)
        self.lib.randReset()
        self.lib.importTrainFiles()
        self.lib.importStreamFiles()
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.tripleTotal = self.lib.getTrainTotal()

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

        #self.graph.init(self.tripleTotal, self.trainList)

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
        self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
        self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
        self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
        self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

        self.stream_seq_size = self.stream_size * (1 + self.negative_ent + self.negative_rel)
        self.stream_h = np.zeros(self.stream_seq_size, dtype=np.int64)
        self.stream_t = np.zeros(self.stream_seq_size, dtype=np.int64)
        self.stream_r = np.zeros(self.stream_seq_size, dtype=np.int64)
        self.stream_y = np.zeros(self.stream_seq_size, dtype=np.float32)


    def corrupt(self, data):
        # data's shape :(batchsize,)
        return [random.randint(0, self.entTotal-1) for i in range(self.batch_size)]

    def sampling(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            0,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h, 
            "batch_t": self.batch_t, 
            "batch_r": self.batch_r, 
            "batch_y": self.batch_y,
            "mode": "normal"
        }

    def sampling_stream(self):
        if self.stream_index == self.nstreams:
            self.stream_index = 0

        index = self.stream_index * self.stream_size 
        self.stream_index += 1

        self.lib.sampling_stream(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            index,
            self.negative_ent,
            self.negative_rel,
            0,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h, 
            "batch_t": self.batch_t, 
            "batch_r": self.batch_r, 
            "batch_y": self.batch_y,
            "mode": "normal"
        }

        '''
        # positive sampling
        for i in range(self.stream_size):
            self.stream_h[i] = self.streamList[index+i,0]
            self.stream_t[i] = self.streamList[index+i,1]
            self.stream_r[i] = self.streamList[index+i,2]

        # negative sampling
        for i in range(1, self.negative_ent + 1):
            neg_index = self.stream_size * i
            rand = random.random()
            if rand > 0.5:
                # corrupt tail
                cor_tail = self.corrupt(self.streamList[index+i,1])
                for ii, neg_data in enumerate(cor_tail):
                    self.stream_t[neg_index+ii] = neg_data
                self.stream_h[neg_index:neg_index+self.stream_size] = self.streamList[index+i,0]
                self.stream_r[neg_index:neg_index+self.stream_size] = self.streamList[index+i,2]
            else :
                # corrupt head
                cor_head = self.corrupt(self.streamList[index+i,0])
                for ii, neg_data in enumerate(cor_head):
                    self.stream_h[neg_index+ii] = neg_data
                self.stream_t[neg_index:neg_index+self.stream_size] = self.streamList[index+i,1]
                self.stream_r[neg_index:neg_index+self.stream_size] = self.streamList[index+i,2]

        return {
            "batch_h": self.stream_h,
            "batch_t": self.stream_t,
            "batch_r": self.stream_r,
            "batch_y": self.stream_y,
            "mode": "normal"
        }
        '''


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

    def get_nstream(self):
        return self.nstreams

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
