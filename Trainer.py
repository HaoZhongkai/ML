import torch as T
import torch.nn as nn
from torch.autograd import variable as V
import numpy as np
from torchnet import meter
from utils import Visualizer
from torch.utils.data import DataLoader
from config import DefaultConfig

class Trainer(object):
    def __init__(self,model=None,opt=DefaultConfig()):
        self.model = model
        self.opt = opt
        self.criterion = opt.criterion
        self.optimizer = opt.optimizer(model.parameters(),lr = opt.lr)


    '''在主函数中定义好Dataset,传进来封装dataloader'''
    def train(self,train_data,val_data=None):
        print('Now we begin training')
        train_dataloader = DataLoader(train_data,batch_size=self.opt.batch_size,shuffle=True)
        #val_dataloader = DataLoader(val_data,self.opt.batch_size,shuffle=True)

        vis = Visualizer(env=self.opt.env)

        if self.opt.use_gpu:
            self.model.cuda()

        previous_loss = 1e10
        loss_meter = meter.AverageValueMeter()
        Confusion_matrix = meter.ConfusionMeter(10)

        for epoch in range(self.opt.max_epoch):
            loss_meter.reset()
            Confusion_matrix.reset()
            for i,(data,label) in enumerate(train_dataloader,0):
                if self.opt.use_gpu:
                    data = data.cuda()
                    label = label.cuda()
                self.optimizer.zero_grad()
                score = self.model(data)
                out_classes = T.argmax(score,1)
                target_digit = T.argmax(label,1)
                loss = self.criterion(score,label)
                loss.backward()
                self.optimizer.step()

                #指标更新
                loss_meter.add(loss.data.cpu())
                Confusion_matrix.add(out_classes,target_digit)
                accuracy = 100*sum(Confusion_matrix.value()[i,i] for i in range(10)
                                   )/Confusion_matrix.value().sum()
                if i % self.opt.print_freq == self.opt.print_freq - 1:
                    print('EPOCH:{0},i:{1},loss:%.6f'.format(epoch, i) %loss.data.cpu
                    ())
                vis.plot('loss', loss_meter.value()[0])
                vis.plot('test_accuracy',accuracy)
            if val_data:
                val_cm,val_ac = self.test(val_data,val=True)
                vis.plot('Val_accuracy',val_ac)
                vis.img('Val Confusion_matrix',T.Tensor(val_cm.value()))

            # 若损失不再下降则降低学习率
            if loss_meter.value()[-1] > previous_loss:
                self.opt.lr = self.opt.lr * self.opt.lr_decay
                print('learning rate:{}'.format(self.opt.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.opt.lr

            previous_loss = loss_meter.value()[-1]

    '''在主函数中封装好Dataset传进来封装dataloader'''
    def test(self,test_data,val=False):
        self.model.eval()
        Confusion_matrix = meter.ConfusionMeter(10)
        test_dataloader = DataLoader(test_data,2000,shuffle=True)

        result = np.array([])
        for i,(data,label) in enumerate(test_dataloader):
            if self.opt.use_gpu:
                data = data.cuda()
                label = label.cuda()
            score = self.model(data)
            target_digit = T.argmax(label,1)
            if not val:
                result = np.concatenate((result,target_digit.cpu().numpy()),0)

            out_digit = T.argmax(score,1)
            Confusion_matrix.add(out_digit,target_digit)

        accuarcy = 100*sum(Confusion_matrix.value()[i,i] for i in range(10)
                           )/Confusion_matrix.value().sum()
        self.model.train()
        if val:
            return Confusion_matrix,accuarcy
        else:
            return result,Confusion_matrix,accuarcy


    '''test_data是输入张量,无需label'''
    def exam(self,test_data):
        self.model.eval()
        score = self.model(test_data)
        out_classes = T.argmax(score,1)
        return out_classes












