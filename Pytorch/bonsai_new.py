from __future__ import print_function, division
import torch
from torch import FloatTensor as FT
from torch import LongTensor as LT
from math import *
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

def ht(A, s):
    A_ = np.copy(A)
    A_ = A_.ravel()
    if len(A_) > 0:
        th = np.percentile(np.abs(A_), (1 - s)*100.0, interpolation='higher')
       	A_[np.abs(A_)<th] = float(0)
    A_ = A_.reshape(A.shape)
    return A_

def cs(src, dest):
    support = np.nonzero(src)
    dest_ = dest
    dest = np.zeros(dest_.shape)
    dest[support] = dest_[support]
    return dest

train_data = []
train_labels = []
X = np.load("/home/cse/phd/anz178419/Datasets/MNIST10/train.npy")
Y = np.load("/home/cse/phd/anz178419/Datasets/MNIST10/test.npy")
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

h = 3
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
        self.h = h; self.int_n = 2**self.h - 1; self.tot_n = 2**(self.h + 1) - 1
        self.pd = pd; self.nf = nf; self.nc = nc

        self.lT = lT; self.lW = lW; self.lV = lV; self.lZ = lZ;
        self.sT = sT; self.sW = sW; self.sV = sV; self.sZ = sZ;

        self.sig = sig
        
        self.Z = nn.Parameter(FT(torch.rand(self.nf, self.pd))-0.5)
        if(self.int_n > 0): self.T = nn.Parameter(torch.FloatTensor(torch.rand(self.int_n, self.pd))-0.5)
        self.V = nn.Parameter(FT(torch.rand(self.tot_n, self.pd, self.nc)) - 0.5)
        self.W = nn.Parameter(FT(torch.rand(self.tot_n, self.pd, self.nc)) - 0.5)

    def forward(self, x):
        batch_size = x.size(0)
        pp = torch.matmul(x,self.Z)/self.pd
        I = Variable(FT(torch.ones(batch_size, self.tot_n)))
        if(self.int_n > 0):
            for i in range(1,self.tot_n):
                j = int(floor((i + 1) / 2) - 1)
                I[:, i] = 0.5 * I[:, j] * (1 + pow(-1, (i + 1) - 2 * (j + 1)) * F.tanh(self.sigI * torch.matmul(pp, self.T[j])))
        score = (torch.matmul(pp, self.W) * F.tanh(self.sig * torch.matmul(pp, self.V)) * torch.t(I).view(self.tot_n, batch_size, 1)).sum(0)
        return score

    def multi_class_loss(self, outputs, labels):
        reg_loss = 0.5 * (self.lW * torch.norm(self.W) + self.lV * torch.norm(self.V) + self.lZ * torch.norm(self.Z))
        if(self.int_n > 0): reg_loss += 0.5 * (self.lT * torch.norm(self.T));
        class_function = nn.CrossEntropyLoss()
        class_loss = class_function(outputs,labels)
        total_loss = reg_loss + class_loss
        return total_loss

bonsai = Bonsai(h,pd,nf,nc,lT,lW,lV,lZ,sT,sW,sV,sZ,sig)
loss_function = lambda x,y: bonsai.multi_class_loss(x,y)
optimizer = optim.SGD(bonsai.parameters(),lr=0.01,momentum=0.9,nesterov=True)
#optimizer = optim.Adadelta(bonsai.parameters(),lr=0.01)

total_epochs = 500
num_iters = int(train_data.shape[0]/batch_size)
total_batches = num_iters * total_epochs
ctr = 0
if bonsai.nc > 2:
	trim_l = 15
else:
	trim_l = 5
iht_done = 0

for i in range(total_epochs):

    for j in range(num_iters):

        if ((ctr == 0) or (ctr == total_batches/3) or (ctr == 2*total_batches/3)):
            bonsai.sigI = 1
            iters_phase = 0

        elif (iters_phase%100 == 0):
            indices = np.random.choice(train_data.shape[0],100)
            batch_x = train_data[indices,:]
            T = bonsai.T.data.numpy()
            Z = bonsai.Z.data.numpy()
            batch_pp = np.matmul(batch_x,Z)
            
            sum_tr = 0.0

            for k in range(0, bonsai.int_n):
                sum_tr = sum_tr + (np.sum(np.abs(np.dot(batch_pp,T[k]))))

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
        inputs, labels = FT(batchx).view(batch_size, nf), LT(batchy).view(batch_size)
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = bonsai(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (ctr >= (total_batches/3) and (ctr < 2*total_batches/3) and ctr%trim_l == 0):
            W_old = bonsai.W.data.numpy(); V_old = bonsai.V.data.numpy(); Z_old = bonsai.Z.data.numpy(); T_old = bonsai.T.data.numpy()
            W_new = ht(W_old, bonsai.sW); V_new = ht(V_old, bonsai.sV); Z_new = ht(Z_old, bonsai.sZ); T_new = ht(T_old, bonsai.sT)
            bonsai.W.data = FT(W_new); bonsai.V.data = FT(V_new); bonsai.Z.data = FT(Z_new); bonsai.T.data = FT(T_new)
            iht_done = 1
        elif ((iht_done == 1 and ctr >= (total_batches/3) and (ctr < 2*total_batches/3) and ctr%trim_l != 0) or (ctr >= 2*total_batches/3)):
            W_old = bonsai.W.data.numpy(); V_old = bonsai.V.data.numpy(); Z_old = bonsai.Z.data.numpy(); T_old = bonsai.T.data.numpy()
            W_new1 = cs(W_new, W_old); V_new1 = cs(V_new, V_old); Z_new1 = cs(Z_new, Z_old); T_new1 = cs(T_new, T_old)
            bonsai.W.data = FT(W_new1); bonsai.V.data = FT(V_new1); bonsai.Z.data = FT(Z_new1); bonsai.T.data = FT(T_new1)
            
        ctr = ctr + 1

    acc = 0
    sigI_old = bonsai.sigI
    bonsai.sigI = 1e9

    batchx = test_data 
    batchy = test_labels 
    inputs, labels = FT(batchx).view(test_data.shape[0],nf), LT(batchy).view(test_data.shape[0])
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = bonsai(inputs)
    _, predictions = torch.max(outputs, 1)
    acc = ((predictions == labels).sum()).data.numpy().tolist()
    end = time()
    bonsai.sigI = sigI_old
    print(float(acc) / Y.__len__())


