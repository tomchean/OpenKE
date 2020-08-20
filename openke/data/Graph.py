import numpy as np

class Graph(object):
    def __init__(self):
        self.head_dict = {}
        self.tail_dict = {} 

    def init(self, entityTotal, trainList):

        # head dic
        head_list = list()
        for i in range(entityTotal):
            head_list.append(trainList[i][0])
        head_key = list(set(head_list))

        self.head_dic = dict()
        for i in head_key:
            self.head_dic[i] = list()
            for j in range(entityTotal):
                if head_list[j] == head_key[i]:
                    self.head_dic[i].append(j)
        
        # tail dic
        tail_list = list()
        for i in range(entityTotal):
            tail_list.append(trainList[i][2])
        tail_key = list(set(tail_list))

        self.tail_dic = dict()
        for i in tail_key:
            self.tail_dic[i] = list()
            for j in range(entityTotal):
                if tail_list[j] == tail_key[i]:
                    self.tail_dic[i].append(j)
        
        # dic to numpy array
        # self.head_dic & self.tail_dic
        for i  in self.head_dic.keys():
            self.head_dic[i] = np.asarray(self.head_dic[i], dtype=np.float32)
        for i  in self.tail_dic.keys():
            self.tail_dic[i] = np.asarray(self.tail_dic[i], dtype=np.float32)
