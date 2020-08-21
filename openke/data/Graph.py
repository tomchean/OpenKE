import numpy as np

class Graph(object):
    def __init__(self):
        self.head_dict = {}
        self.tail_dict = {} 

    def init(self, entityTotal, trainList):

        tupleTotal = len(trainList)

        # head dic
        head_list = list()
        for i in range(tupleTotal):
            head_list.append(trainList[i][0])
        head_key = [x for i, x in enumerate(head_list) if head_list.index(x) == i]

        self.head_dic = dict()
        for i in range(len(head_key)):
            self.head_dic[head_key[i]] = list()
            for j in range(tupleTotal):
                if head_list[j] == head_key[i]:
                    self.head_dic[head_key[i]].append(trainList[j])
        
        # tail dic
        tail_list = list()
        for i in range(tupleTotal):
            tail_list.append(trainList[i][2])
        tail_key = [x for i, x in enumerate(tail_list) if tail_list.index(x) == i]

        self.tail_dic = dict()
        for i in range(len(tail_key)):
            self.tail_dic[tail_key[i]] = list()
            for j in range(tupleTotal):
                if tail_list[j] == tail_key[i]:
                    self.tail_dic[tail_key[i]].append(trainList[j])
        
        # dic to numpy array
        # self.head_dic & self.tail_dic
        for i  in self.head_dic.keys():
            self.head_dic[i] = np.asarray(self.head_dic[i], dtype=np.float32)
        for i  in self.tail_dic.keys():
            self.tail_dic[i] = np.asarray(self.tail_dic[i], dtype=np.float32)
