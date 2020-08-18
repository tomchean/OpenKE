
class Graph(object):
    def __init__(self):
        self.head_dict = {}
        self.tail_dict = {} 

    def init(self, entityTotal, trainList):
        self.head_dict = { i : [] for i in range(entityTotal)}
        self.tail_dict = { i : [] for i in range(entityTotal)}

        for data in trainList:
            self.head_dict[data[0]].append((data[1], data[2]))
            self.tail_dict[data[1]].append((data[0], data[2]))
