from Model import CNN_Cifar
from config import DefaultConfig
from data import Cifar
from Trainer import Trainer
import matplotlib.pyplot as plt

opt = DefaultConfig()
Train_data_root = 'E:\Machine Learning\DATASET\Cifar10\\Train.pkl'
Test_data_root = 'E:\Machine Learning\DATASET\Cifar10\\Test.pkl'

Train_dataset = Cifar(Train_data_root,train=True)
Val_dataset = Cifar(Train_data_root,train=False)
Test_dataset = Cifar(Test_data_root,train=True)

Model = CNN_Cifar()
Model.cuda()
TrainerCifar = Trainer(Model,opt)

TrainerCifar.train(train_data=Train_dataset,val_data=Val_dataset)
results,Confusion_matrix,accuracy = TrainerCifar.test(Test_dataset,val=False)
plt.imshow(Confusion_matrix.value())
print(accuracy)
Model.save(opt.save_model_path)

