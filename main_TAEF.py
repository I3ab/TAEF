import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
import net_TAEF
import h5py
import random


def loadData():

    pavia = h5py.File('F:/code4/TAEF/pavia_split.mat', 'r')
    groundtruth = np.transpose(pavia['groundtruth'][:])
    datatrain = np.transpose(pavia['data_train'][:])
    data = np.transpose(pavia['data'][:])
    datatest = np.transpose(pavia['data_test'][:])

    # MUUFL = h5py.File('F:/code4/TAEF/MUUFL_split.mat', 'r')
    # groundtruth = np.transpose(MUUFL['groundtruth'][:])
    # datatrain = np.transpose(MUUFL['data_train'][:])
    # data = np.transpose(MUUFL['data'][:])
    # datatest = np.transpose(MUUFL['data_test'][:])

    return datatrain, groundtruth,data,datatest

def train(data,lr, epochs):
    Bands = np.size(data, 2)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = net_TAEF.net_TAEF().to(device)
    optimizer = optim.Adam(net.parameters(), lr)
    total_loss = 0

    data=torch.from_numpy(data)
    data = data.to(torch.float32)

    loss_save = torch.zeros(epochs*19)
    loss_save_epoch=torch.zeros(epochs)

    for epoch in range(epochs):
        tic_train = time.perf_counter()
        net.train()

        index=[iindex for iindex in range(361)]
        random.shuffle(index)

        loss_epoch=0
        for i in range(19):
            loss_batch = 0
            for j in range(19):
                data0 = data[index[i * 19 + j]]
                dataij_2D = data0
                dataij_2D, target = dataij_2D.to(device),dataij_2D.to(device)
                p,_, outputs = net(dataij_2D)
                diff = outputs - target
                loss1=diff.norm(2)
                loss_batch+=loss1
                loss_epoch += loss1

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            loss_save[epoch*19+i] = loss_batch

        time_epoch = time.perf_counter() - tic_train
        loss_save_epoch[epoch]=loss_epoch
        print('[epoch:%.1f] [loss：%.4f] [loss_epoch：%.4f] [time_epoch：%.4f]'% (epoch+1,loss_batch,loss_epoch,time_epoch))

        if loss_epoch < 200:  # pavia
        # if loss_epoch < 800:  # MUUFL
            break


    print('Finished Training')

    return net, device, loss_save,loss_save_epoch

def test(device, net, datatest,data):
    net.eval()
    Bands=np.size(data,2)

    datatest = torch.from_numpy(datatest)
    datatest = datatest.to(torch.float32)
    result=torch.zeros(100,100,Bands)

    for i in range(361):
        data0 = datatest[i]
        dataij_2D = data0
        dataij_2D = dataij_2D.to(device)
        z,_, outputs = net(dataij_2D)
        outputs_3D=outputs.reshape(10,10,Bands)
        outputs_3D=outputs_3D.transpose(0,1)

        row=i//19
        col=i%19

        result[5*row:5*row+10,5*col:5*col+10]+=outputs_3D

    for i in range(100):
        for j in range(100):
            if i<5 or i>94:
                if 4<j<95:
                    result[i,j]/=2
            else:
                if j<5 or j>94:
                    result[i, j] /= 2
                else:
                    result[i, j] /= 4


    result2 = [0 for i in range(10000)]
    data = torch.from_numpy(data)
    data = data.to(torch.float32)
    for n in range(10000):
        diff_n=result[n//100,n%100]-data[n//100,n%100]
        n2=diff_n.norm(2)
        result2[n] = n2.item()

    mi=min(result2)
    ma=max(result2)
    di=ma-mi
    result2=[(i-mi)/di for i in result2]

    result2=np.array(result2)
    result2D=np.resize(result2,[100,100])
    return result2D,result

if __name__ == '__main__':
    datatrain,groundtruth,data,datatest=loadData()

    tic1 = time.perf_counter()
    lr = 0.005
    epochs = 20
    net, device, loss_save,loss_save_epoch = train(datatrain, lr, epochs)
    result2D, reconstruction = test(device, net, datatest,data)
    time = time.perf_counter()-tic1

    total_pixels=10000
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(np.squeeze(np.resize(groundtruth, [total_pixels, 1])),
                                     np.squeeze(np.resize(result2D, [total_pixels, 1])))
    roc_auc = auc(fpr, tpr)
    print('[AUC: %.4f]' %(roc_auc))
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(result2D)
    plt.subplot(1,2,2)
    plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC curve (area = %0.5f)' % roc_auc)
    plt.xlim([0,1])
    plt.xlim([0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating charateristic')
    plt.legend(loc="lower right")
    plt.show(block=True)

    sio.savemat('F:/code4/TAEF/save/pavia_rec.mat',mdict={'result': result2D, 'time': time, 'AUC': roc_auc, 'loss_save': loss_save.detach().numpy(),'lr': lr,'reconstruction':reconstruction.detach().numpy(), 'epochs': epochs, 'loss_save_epoch':loss_save_epoch.detach().numpy(),})
