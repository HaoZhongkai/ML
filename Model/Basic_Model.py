import torch.nn as nn
import time
import os
import torch

'''该Basic Model不提供训练与测试方法,而考虑将其封装为trainer对象方便在主函数中查看数据'''
class Basic_Model(nn.Module):
    def __init__(self):
        super(Basic_Model,self).__init__()
        self.Model_name = str(type(self))

    def load(self,path):
        self.load_state_dict(path)

    '''name需要加后缀'''
    def save(self,path,name=None):
        if name is None:
            name = time.strftime('%m%d_%H_%M.pkl')
            name = path + name
        torch.save(self.state_dict(),name)



