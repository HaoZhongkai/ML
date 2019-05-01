import os
import warnings
import torch as T
import torch.optim
import torch.nn as nn

class DefaultConfig(object):
    env = 'default'     #visdom环境
    model = 'CNN'   #模型
    data_root = 'E:\Machine Learning\DATASET\Cifar10\Train.pkl'
    load_model_path = None #'E:\Machine Learning\CNN_minist'
    save_model_path = 'E:\Machine Learning\Cifar10\Model\model_file'
    batch_size = 50
    use_gpu = True
    num_workers = 1
    print_freq = 20     #打印info

    result_file = 'result.csv'

    max_epoch = 13
    lr = 1
    lr_decay = 0.95       #当 val_loss增加时,lr*=lr_decay
    weight_decay = 1e-4
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD

    def parse(self,kwargs):
        '''根据字典更新config'''
        if not kwargs:
            for k,v in kwargs.items():
                if not hasattr(self,k):
                    warnings.warn("警告:opt没有该属性%s" %k)
                setattr(self,k,v)
            print('设置:')
            for k,v in self.__class__.__dict__.items():
                if not k.startswith('__'):
                    print(k,getattr(self,k))
