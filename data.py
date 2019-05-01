import numpy as np
from torch.utils import data
import pickle
import torch


'''该数据集直接用data创建，读取data留给主函数操作'''
'''Data封成list,为(N*data,N*label)方便读取'''
class Cifar(data.Dataset):
    def __init__(self,PATH,transform=None,train=True):
        '''获取并划分数据'''
        Data = pickle.load(open(PATH,'rb'))
        self.train = train
        num = len(Data[0])
        if self.train:
            self.data = (Data[0][:int(0.2*num)],Data[1][:int(0.2*num)])
        else:
            self.data = (Data[0][int(0.8*num):],Data[1][int(0.8*num):])

    def __getitem__(self, index):
        data = torch.Tensor(self.data[0][index])
        label = torch.Tensor(self.data[1][index])
        return data,label

    def __len__(self):
        return len(self.data[0])
