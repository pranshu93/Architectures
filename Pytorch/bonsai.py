from __future__ import print_function, division
import torch
from math import *
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
import torch.nn.functional as F

def hard_thrsd(A, s):
    A_ = np.copy(A)
    A_ = A_.ravel()
    if len(A_) > 0:
	th = np.percentile(np.abs(A_), (1 - s)*100.0, interpolation='higher')
       	A_[np.abs(A_)<th] = float(0)
    A_ = A_.reshape(A.shape)
    return A_

def copy_support(src, dest):
    support = np.nonzero(src)
    dest_ = dest
    dest = np.zeros(dest_.shape)
    dest[support] = dest_[support]
    return dest

train_data = []
train_labels = []
X = np.load("train.npy")
Y = np.load("test.npy")
X = np.concatenate((X,np.ones((X.__len__(),1))),axis=1)
Y = np.concatenate((Y,np.ones((Y.__len__(),1))),axis=1)
train_data = X[:,1:]
train_labels = X[:,0]
test_data = Y[:,1:]
test_labels = Y[:,0]

mean=np.mean(train_data,0)
std=np.std(train_data,0)
std[std[:]<0.00001]=1
train_data=(train_data-mean)/std
test_data=(test_data-mean)/std

h = 4
pd = 20
nf = int(train_data.shape[1])
nc = int(np.max(train_labels)) - int(np.min(train_labels)) + 1
train_labels = np.array(train_labels - np.min(train_labels),dtype=int)
test_labels = np.array(test_labels - np.min(test_labels),dtype=int)

lT = 1e-2
lW = 1e-2
lV = 1e-2
lZ = 1e-3
sT = 0.5
sW = 0.5
sV = 0.5
sZ = 0.1
batch_size = 100
sig = 4

class Bonsai(nn.Module):
    def __init__(self,h,pd,nf,nc,lT,lW,lV,lZ,sT,sW,sV,sZ,sig):
        super(Bonsai, self).__init__()
        self.h = h
        self.int_n = 2**self.h - 1
        self.tot_n = 2**(self.h + 1) - 1
        self.pd = pd
        self.nf = nf

        self.nc = nc

        self.lT = lT
        self.lW = lW
        self.lV = lV
        self.lZ = lZ

        self.sT = sT
        self.sW = sW
        self.sV = sV
        self.sZ = sZ

        self.sig = sig
        self.sigI = 1
        self.Z = nn.Parameter(torch.FloatTensor(torch.rand(self.pd, self.nf))-0.5)

        if(self.int_n > 0):
            self.T = nn.Parameter(torch.FloatTensor(torch.rand(self.int_n, self.pd, 1))-0.5)

        self.V = nn.Parameter(torch.FloatTensor(torch.rand(self.tot_n, self.pd, self.nc)) - 0.5)
        self.W = nn.Parameter(torch.FloatTensor(torch.rand(self.tot_n, self.pd, self.nc)) - 0.5)

    def forward(self, x):
        batch_size = x.size(0)
        pp = torch.matmul(self.Z,x.view(batch_size,self.nf,1))/self.pd
        pp = pp.view(batch_size,self.pd)

        I = Variable(torch.FloatTensor(torch.ones(batch_size, self.tot_n)))
        score = Variable(torch.FloatTensor(torch.zeros(batch_size, self.nc)))

        if(self.int_n > 0):
            for i in range(1,self.tot_n):
                j = int(floor((i + 1) / 2) - 1)
                I[:, i] = 0.5 * I[:, j] * (1 + pow(-1, (i + 1) - 2 * (j + 1)) * F.tanh(self.sigI * torch.t(torch.matmul(pp, self.T[j]))))

        for i in range(self.tot_n):
            score = score + torch.t(torch.t(torch.matmul(pp, self.W[i]) * F.tanh(self.sig * torch.matmul(pp, self.V[i]))) * I[:, i])
        return score

    def multi_class_loss(self, outputs, labels):
        reg_loss = 0.5 * (self.lW * torch.norm(self.W) + self.lV * torch.norm(self.V) + self.lZ * torch.norm(self.Z))
        if(self.int_n > 0):
            reg_loss += 0.5 * (self.lT * torch.norm(self.T))
        class_function = nn.CrossEntropyLoss()
        class_loss = class_function(outputs,labels)
        total_loss = reg_loss + class_loss
        return total_loss

bonsai = Bonsai(h,pd,nf,nc,lT,lW,lV,lZ,sT,sW,sV,sZ,sig)
loss_function = lambda x,y: bonsai.multi_class_loss(x,y)
optimizer = optim.Adam(bonsai.parameters(),lr=0.005)

total_epochs = 10
num_iters = int(train_data.shape[0]/batch_size)
total_batches = num_iters * total_epochs
counter = 0
if bonsai.nc > 2:
	trimlevel = 15
else:
	trimlevel = 5
iht_done = 0

for i in range(total_epochs):

    for j in range(num_iters):

        if ((counter == 0) or (counter == total_batches/3) or (counter == 2*total_batches/3)):
            bonsai.sigI = 1
            iters_phase = 0

        elif (iters_phase%100 == 0):
            indices = np.random.choice(train_data.shape[0],100)
            batch_x = train_data[indices,:]
            T = bonsai.T.data.numpy()
            Z = bonsai.Z.data.numpy()
            batch_pp = np.matmul(Z,np.transpose(batch_x))
            
            sum_tr = 0.0

            for k in range(0, bonsai.int_n):
                sum_tr = sum_tr + (np.sum(np.abs(np.dot(np.transpose(T[k]), batch_pp))))

            if(bonsai.int_n > 0):
                sum_tr = sum_tr/(100*bonsai.int_n)
                sum_tr = 0.1/sum_tr
            else:
                sum_tr = 0.1
            sum_tr = min(1000,sum_tr*(2**(float(iters_phase)/(float(total_batches)/30.0))))

            bonsai.sigI = float(sum_tr)

        iters_phase = iters_phase + 1

        optimizer.zero_grad()
        batchx = train_data[j * batch_size:(j + 1) * batch_size, :]
        batchy = train_labels[j * batch_size:(j + 1) * batch_size]
        inputs, labels = torch.FloatTensor(batchx).view(batch_size, nf), torch.LongTensor(batchy).view(batch_size)
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = bonsai(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (counter >= (total_batches/3) and (counter < 2*total_batches/3) and counter%trimlevel == 0):
            
            W_old = bonsai.W.data.numpy()
            V_old = bonsai.V.data.numpy()
            Z_old = bonsai.Z.data.numpy()
            T_old = bonsai.T.data.numpy()

            W_new = hard_thrsd(W_old, bonsai.sW)
            V_new = hard_thrsd(V_old, bonsai.sV)
            Z_new = hard_thrsd(Z_old, bonsai.sZ)
            T_new = hard_thrsd(T_old, bonsai.sT)
            

            bonsai.W.data = torch.FloatTensor(W_new)
            bonsai.V.data = torch.FloatTensor(V_new)
            bonsai.Z.data = torch.FloatTensor(Z_new)
            bonsai.T.data = torch.FloatTensor(T_new)

            iht_done = 1
        elif ((iht_done == 1 and counter >= (total_batches/3) and (counter < 2*total_batches/3) and counter%trimlevel != 0) or (counter >= 2*total_batches/3)):

            W_old = bonsai.W.data.numpy()
            V_old = bonsai.V.data.numpy()
            Z_old = bonsai.Z.data.numpy()
            T_old = bonsai.T.data.numpy()

            W_new1 = copy_support(W_new, W_old)
            V_new1 = copy_support(V_new, V_old)
            Z_new1 = copy_support(Z_new, Z_old)
            T_new1 = copy_support(T_new, T_old)

            bonsai.W.data = torch.FloatTensor(W_new1)
            bonsai.V.data = torch.FloatTensor(V_new1)
            bonsai.Z.data = torch.FloatTensor(Z_new1)
            bonsai.T.data = torch.FloatTensor(T_new1)
            
        counter = counter + 1

    acc = 0
    sigI_old = bonsai.sigI
    bonsai.sigI = 1e9

    for j in range(int(test_data.__len__() / batch_size)):
        batchx = test_data[j * batch_size:(j + 1) * batch_size, :]
        batchy = test_labels[j * batch_size:(j + 1) * batch_size]       
        inputs, labels = torch.FloatTensor(batchx).view(batch_size,nf), torch.LongTensor(batchy).view(batch_size)
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = bonsai(inputs)
        _, predictions = torch.max(outputs, 1)
        acc += ((predictions == labels).sum()).data.numpy()[0]
    end = time()
    bonsai.sigI = sigI_old
    print("Test Accuracy after Iteration", i + 1, "=", end=' ')
    print(float(100 * acc) / Y.__len__(), "%")


