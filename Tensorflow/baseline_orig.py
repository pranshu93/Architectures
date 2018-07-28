from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import timeit
import os
from thread import start_new_thread
import sys
pid = os.getpid()
def hard_thrsd(A, s):
    A_ = np.copy(A)
    A_ = A_.ravel()
    if len(A_) > 0:
        th = np.percentile(np.abs(A_), (1 - s)*100.0, interpolation='higher')
        A_[np.abs(A_)<th] = 0.0
    A_ = A_.reshape(A.shape)
    return A_

def copy_support(src, dest):
    support = np.nonzero(src)
    dest_ = dest
    dest = np.zeros(dest_.shape)
    dest[support] = dest_[support]
    return dest

tf.set_random_seed(42)
np.random.seed(42)


data_dir=sys.argv[1]
inp_dims=int(sys.argv[2])
num_classes = int(sys.argv[3])
train=np.load(data_dir+'train.npy')
test=np.load(data_dir+'test.npy')
train_feats=train[:,1:inp_dims+1]
train_lbl=train[:,0]
test_feats=test[:,1:inp_dims+1]
test_lbl=test[:,0]
# remove this if indexing from zero

##preprocessing 
mean=np.mean(train_feats,0)
std=np.std(train_feats,0)
std[std[:]<0.00001]=1
train_feats=(train_feats-mean)/std

#mean=np.mean(test_feats,0)
#std=np.std(test_feats,0)
test_feats=(test_feats-mean)/std

lab=train_lbl.astype('uint8')
lab=np.array(lab) - min(lab)
lab1= np.zeros((train_feats.shape[0], num_classes))
lab1[np.arange(train_feats.shape[0]), lab] = 1
train_labels=lab1

lab=test_lbl.astype('uint8')
lab=np.array(lab) - min(lab)
lab1= np.zeros((test_feats.shape[0], num_classes))
lab1[np.arange(test_feats.shape[0]), lab] = 1
test_labels=lab1

# Training Parameters
learning_rate = 0.01
training_steps = 100
batch_size = 100
display_step = 200

# Network Parameters
num_input = int(sys.argv[4])#512 # MNIST data input (img shape: 28*28)
timesteps =  int(sys.argv[5])#16# timesteps
num_hidden =int(sys.argv[6])#16#hidden layer num of features

#W = int(os.popen("pidstat -r -p " + str(pid)).read().splitlines()[3].split()[-3])  
#print("Memory:"+str(W))

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])
lr = 0.01
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
class rnn:
    def __init__(self,num_timesteps,dataDimensions,projectionDimensions):

        self.num_timesteps=num_timesteps
        self.dataDimensions=dataDimensions
        self.projectionDimensions=projectionDimensions
        self.W = []
        self.U = []
        self.B = []
        self.sW = 1.0
        self.sU = 1.0
        self.sB = 1.0

        W_ini = tf.random_normal([dataDimensions, projectionDimensions],0.0,0.1)
        self.W.append(tf.Variable(W_ini, name='W', dtype=tf.float32))

        U_ini = tf.random_normal([projectionDimensions,projectionDimensions],0.0,0.1)
        self.U.append(tf.Variable(U_ini, name='U', dtype=tf.float32))

        B_ini = np.zeros([1,projectionDimensions])  ##try with ones initialization
        self.B.append(tf.Variable(B_ini, name='B', dtype=tf.float32))

    def placeholder(self):
        self.Wth = []
        self.Uth = []
        self.Bth = []
        for i in range(len(self.W)):
            self.Wth.append(tf.placeholder(tf.float32, name='Wth'+str(i)))
            self.Uth.append(tf.placeholder(tf.float32, name='Uth'+str(i)))
            self.Bth.append(tf.placeholder(tf.float32, name='Bth'+str(i)))

        self.Wst = []
        self.Ust = []
        self.Bst = []
        for i in range(len(self.W)):
            self.Wst.append(tf.placeholder(tf.float32, name='Wst'+str(i)))
            self.Ust.append(tf.placeholder(tf.float32, name='Ust'+str(i)))
            self.Bst.append(tf.placeholder(tf.float32, name='Bst'+str(i)))

        self.hardThrsd()
        self.sparseTraining()

    def hardThrsd(self):
        self.W_op1 = []
        self.U_op1 = []
        self.B_op1 = []
        for i in range(len(self.W)):
            self.W_op1.append(self.W[i].assign(self.Wth[i]))
            self.U_op1.append(self.U[i].assign(self.Uth[i]))
            self.B_op1.append(self.B[i].assign(self.Bth[i]))
            if i == 0:
                self.hard_thrsd_grp = tf.group(self.W_op1[i], self.U_op1[i], self.B_op1[i])
            else:
                self.hard_thrsd_grp = tf.group(self.W_op1[i], self.U_op1[i], self.B_op1[i], self.hard_thrsd_grp)

    def sparseTraining(self):
        self.W_op2 = []
        self.U_op2 = []
        self.B_op2 = []
        for i in range(len(self.W)):
            self.W_op2.append(self.W[i].assign(self.Wst[i]))
            self.U_op2.append(self.U[i].assign(self.Ust[i]))
            self.B_op2.append(self.B[i].assign(self.Bst[i]))
            if i == 0:
                self.sparse_retrain_grp = tf.group(self.W_op2[i], self.U_op2[i], self.B_op2[i])
            else:
                self.sparse_retrain_grp = tf.group(self.W_op2[i], self.U_op2[i], self.B_op2[i], self.sparse_retrain_grp)

    def runHardThrsd(self, sess):
        self.W_old = []
        self.U_old = []
        self.B_old = []

        self.W_new = []
        self.U_new = []
        self.B_new = []

        for i in range(len(self.W)):
            self.W_old.append(self.W[i].eval())
            self.U_old.append(self.U[i].eval())
            self.B_old.append(self.B[i].eval())

            self.W_new.append(hard_thrsd(self.W_old[i], self.sW))
            self.U_new.append(hard_thrsd(self.U_old[i], self.sU))
            self.B_new.append(hard_thrsd(self.B_old[i], self.sB))

        fd_thrsd = {}
        for i in range(len(self.W)):
            fd_thrsd[self.Wth[i]] = self.W_new[i]
            fd_thrsd[self.Uth[i]] = self.U_new[i]
            fd_thrsd[self.Bth[i]] = self.B_new[i]

        sess.run(self.hard_thrsd_grp, feed_dict=fd_thrsd)


    def runSparseTraining(self, sess):
        self.W_old = []
        self.U_old = []
        self.B_old = []

        self.W_new1 = []
        self.U_new1 = []
        self.B_new1 = []

        for i in range(len(self.W)):
            self.W_old.append(self.W[i].eval())
            self.U_old.append(self.U[i].eval())
            self.B_old.append(self.B[i].eval())

            self.W_new1.append(copy_support(self.W_new[i], self.W_old[i]))
            self.U_new1.append(copy_support(self.U_new[i], self.U_old[i]))
            self.B_new1.append(copy_support(self.B_new[i], self.B_old[i]))

        fd_st = {}
        for i in range(len(self.W)):
            fd_st[self.Wst[i]] = self.W_new1[i]
            fd_st[self.Ust[i]] = self.U_new1[i]
            fd_st[self.Bst[i]] = self.B_new1[i]

        sess.run(self.sparse_retrain_grp, feed_dict=fd_st)


    def compute(self,x):
        state=[]
        for i in range(self.num_timesteps):
            # print(x[0])    
            if(i==0):
                state=tf.tanh(tf.matmul(x[i], self.W[0])+self.B)   ##no nonlinearity better results 
            else:            
                state=tf.tanh(tf.matmul(x[i], self.W[0])+tf.matmul(state,self.U[0])+self.B[0])
            state = tf.reshape(state, [-1, self.projectionDimensions]) 
        return state

class gru:
    def __init__(self,num_timesteps, dataDimensions, projectionDimensions):

        self.dataDimensions=dataDimensions
        self.num_timesteps=num_timesteps
        self.projectionDimensions=projectionDimensions
        self.W = []
        self.U = []
        self.B = []
        self.sW = 1.0
        self.sU = 1.0
        self.sB = 1.0

        Wr_ini = tf.random_normal([dataDimensions, projectionDimensions],0.0,0.1)
        self.W.append(tf.Variable(Wr_ini, name='Wr', dtype=tf.float32))

        Ur_ini = tf.random_normal([projectionDimensions,projectionDimensions],0.0,0.1)
        self.U.append(tf.Variable(Ur_ini, name='Ur', dtype=tf.float32))

        Br_ini = np.ones([1,projectionDimensions])
        self.B.append(tf.Variable(Br_ini, name='Br', dtype=tf.float32))
        
        ##Update Gate
        Wz_ini = tf.random_normal([dataDimensions, projectionDimensions],0.0,0.1)
        self.W.append(tf.Variable(Wz_ini, name='Wz', dtype=tf.float32))

        Uz_ini = tf.random_normal([projectionDimensions,projectionDimensions],0.0,0.1)
        self.U.append(tf.Variable(Uz_ini, name='Uz', dtype=tf.float32))

        Bz_ini = np.ones([1,projectionDimensions])
        self.B.append(tf.Variable(Bz_ini, name='Bz', dtype=tf.float32))

        ##
        W_ini = tf.random_normal([dataDimensions, projectionDimensions],0.0,0.1)
        self.W.append(tf.Variable(W_ini, name='W', dtype=tf.float32))

        U_ini = tf.random_normal([projectionDimensions,projectionDimensions],0.0,0.1)
        self.U.append(tf.Variable(U_ini, name='U', dtype=tf.float32))

        B_ini = np.ones([1,projectionDimensions])   ##in true implementation it is zero but 1 seems to be working well
        self.B.append(tf.Variable(B_ini, name='B', dtype=tf.float32))

    def placeholder(self):
        self.Wth = []
        self.Uth = []
        self.Bth = []
        for i in range(len(self.W)):
            self.Wth.append(tf.placeholder(tf.float32, name='Wth'+str(i)))
            self.Uth.append(tf.placeholder(tf.float32, name='Uth'+str(i)))
            self.Bth.append(tf.placeholder(tf.float32, name='Bth'+str(i)))

        self.Wst = []
        self.Ust = []
        self.Bst = []
        for i in range(len(self.W)):
            self.Wst.append(tf.placeholder(tf.float32, name='Wst'+str(i)))
            self.Ust.append(tf.placeholder(tf.float32, name='Ust'+str(i)))
            self.Bst.append(tf.placeholder(tf.float32, name='Bst'+str(i)))

        self.hardThrsd()
        self.sparseTraining()

    def hardThrsd(self):
        self.W_op1 = []
        self.U_op1 = []
        self.B_op1 = []
        for i in range(len(self.W)):
            self.W_op1.append(self.W[i].assign(self.Wth[i]))
            self.U_op1.append(self.U[i].assign(self.Uth[i]))
            self.B_op1.append(self.B[i].assign(self.Bth[i]))
            if i == 0:
                self.hard_thrsd_grp = tf.group(self.W_op1[i], self.U_op1[i], self.B_op1[i])
            else:
                self.hard_thrsd_grp = tf.group(self.W_op1[i], self.U_op1[i], self.B_op1[i], self.hard_thrsd_grp)

    def sparseTraining(self):
        self.W_op2 = []
        self.U_op2 = []
        self.B_op2 = []
        for i in range(len(self.W)):
            self.W_op2.append(self.W[i].assign(self.Wst[i]))
            self.U_op2.append(self.U[i].assign(self.Ust[i]))
            self.B_op2.append(self.B[i].assign(self.Bst[i]))
            if i == 0:
                self.sparse_retrain_grp = tf.group(self.W_op2[i], self.U_op2[i], self.B_op2[i])
            else:
                self.sparse_retrain_grp = tf.group(self.W_op2[i], self.U_op2[i], self.B_op2[i], self.sparse_retrain_grp)

    def runHardThrsd(self, sess):
        self.W_old = []
        self.U_old = []
        self.B_old = []

        self.W_new = []
        self.U_new = []
        self.B_new = []

        for i in range(len(self.W)):
            self.W_old.append(self.W[i].eval())
            self.U_old.append(self.U[i].eval())
            self.B_old.append(self.B[i].eval())

            self.W_new.append(hard_thrsd(self.W_old[i], self.sW))
            self.U_new.append(hard_thrsd(self.U_old[i], self.sU))
            self.B_new.append(hard_thrsd(self.B_old[i], self.sB))

        fd_thrsd = {}
        for i in range(len(self.W)):
            fd_thrsd[self.Wth[i]] = self.W_new[i]
            fd_thrsd[self.Uth[i]] = self.U_new[i]
            fd_thrsd[self.Bth[i]] = self.B_new[i]

        sess.run(self.hard_thrsd_grp, feed_dict=fd_thrsd)


    def runSparseTraining(self, sess):
        self.W_old = []
        self.U_old = []
        self.B_old = []

        self.W_new1 = []
        self.U_new1 = []
        self.B_new1 = []

        for i in range(len(self.W)):
            self.W_old.append(self.W[i].eval())
            self.U_old.append(self.U[i].eval())
            self.B_old.append(self.B[i].eval())

            self.W_new1.append(copy_support(self.W_new[i], self.W_old[i]))
            self.U_new1.append(copy_support(self.U_new[i], self.U_old[i]))
            self.B_new1.append(copy_support(self.B_new[i], self.B_old[i]))

        fd_st = {}
        for i in range(len(self.W)):
            fd_st[self.Wst[i]] = self.W_new1[i]
            fd_st[self.Ust[i]] = self.U_new1[i]
            fd_st[self.Bst[i]] = self.B_new1[i]

        sess.run(self.sparse_retrain_grp, feed_dict=fd_st)

    def compute(self,x):

        state=[]
        for i in range(self.num_timesteps):
            #print(x[0])    
            if(i==0):
                r=tf.sigmoid(tf.matmul(x[i], self.W[0])+self.B[0])
                z=tf.sigmoid(tf.matmul(x[i], self.W[1])+self.B[1])
                h=tf.tanh(tf.matmul(x[i], self.W[2])+self.B[2])
                state=tf.multiply(h,tf.subtract(1.0,z))
                print(state)
            else:            
                r=tf.sigmoid(tf.matmul(x[i], self.W[0])+tf.matmul(state,self.U[0])+self.B[0])
                z=tf.sigmoid(tf.matmul(x[i], self.W[1])+tf.matmul(state,self.U[1])+self.B[1])
                #print(tf.multiply(r,state))
                h=tf.tanh(tf.matmul(x[i], self.W[2])+tf.matmul(tf.multiply(r,state),self.U[2])+self.B[2])
                state=tf.multiply(state,z)+tf.multiply(h,tf.subtract(1.0,z))
        
        return state 
                
class lstm:
    def __init__(self,num_timesteps,dataDimensions,projectionDimensions):

        self.num_timesteps=num_timesteps
        self.dataDimensions=dataDimensions
        self.projectionDimensions=projectionDimensions
        self.W = []
        self.U = []
        self.B = []
        self.sW = 1.0
        self.sU = 1.0
        self.sB = 1.0

        Wi_ini = tf.random_normal([dataDimensions, projectionDimensions],0.0,0.1)
        self.W.append(tf.Variable(Wi_ini, name='Wi', dtype=tf.float32))

        Ui_ini = tf.random_normal([projectionDimensions,projectionDimensions],0.0,0.1)
        self.U.append(tf.Variable(Ui_ini, name='Ui', dtype=tf.float32))

        Bi_ini = np.ones([1,projectionDimensions])
        self.B.append(tf.Variable(Bi_ini, name='Bi', dtype=tf.float32))
        
        ##forget Gate
        Wf_ini = tf.random_normal([dataDimensions, projectionDimensions],0.0,0.1)
        self.W.append(tf.Variable(Wf_ini, name='Wf', dtype=tf.float32))

        Uf_ini = tf.random_normal([projectionDimensions,projectionDimensions],0.0,0.1)
        self.U.append(tf.Variable(Uf_ini, name='Uf', dtype=tf.float32))

        Bf_ini = np.ones([1,projectionDimensions])
        self.B.append(tf.Variable(Bf_ini, name='Bf', dtype=tf.float32))

        ##output gate
        Wo_ini = tf.random_normal([dataDimensions, projectionDimensions],0.0,0.1)
        self.W.append(tf.Variable(Wo_ini, name='Wo', dtype=tf.float32))

        Uo_ini = tf.random_normal([projectionDimensions,projectionDimensions],0.0,0.1)
        self.U.append(tf.Variable(Uo_ini, name='Uo', dtype=tf.float32))

        Bo_ini = np.ones([1,projectionDimensions])
        self.B.append(tf.Variable(Bo_ini, name='Bo', dtype=tf.float32))

        ##
        Wg_ini = tf.random_normal([dataDimensions, projectionDimensions],0.0,0.1)
        self.W.append(tf.Variable(Wg_ini, name='Wg', dtype=tf.float32))

        Ug_ini = tf.random_normal([projectionDimensions,projectionDimensions],0.0,0.1)
        self.U.append(tf.Variable(Ug_ini, name='Ug', dtype=tf.float32))

        Bg_ini = np.ones([1,projectionDimensions])
        self.B.append(tf.Variable(Bg_ini, name='Bg', dtype=tf.float32))

    def placeholder(self):
        self.Wth = []
        self.Uth = []
        self.Bth = []
        for i in range(len(self.W)):
            self.Wth.append(tf.placeholder(tf.float32, name='Wth'+str(i)))
            self.Uth.append(tf.placeholder(tf.float32, name='Uth'+str(i)))
            self.Bth.append(tf.placeholder(tf.float32, name='Bth'+str(i)))

        self.Wst = []
        self.Ust = []
        self.Bst = []
        for i in range(len(self.W)):
            self.Wst.append(tf.placeholder(tf.float32, name='Wst'+str(i)))
            self.Ust.append(tf.placeholder(tf.float32, name='Ust'+str(i)))
            self.Bst.append(tf.placeholder(tf.float32, name='Bst'+str(i))) 

        self.hardThrsd()
        self.sparseTraining()

    def hardThrsd(self):
        self.W_op1 = []
        self.U_op1 = []
        self.B_op1 = []
        for i in range(len(self.W)):
            self.W_op1.append(self.W[i].assign(self.Wth[i]))
            self.U_op1.append(self.U[i].assign(self.Uth[i]))
            self.B_op1.append(self.B[i].assign(self.Bth[i]))
            if i == 0:
                self.hard_thrsd_grp = tf.group(self.W_op1[i], self.U_op1[i], self.B_op1[i])
            else:
                self.hard_thrsd_grp = tf.group(self.W_op1[i], self.U_op1[i], self.B_op1[i], self.hard_thrsd_grp)

    def sparseTraining(self):
        self.W_op2 = []
        self.U_op2 = []
        self.B_op2 = []
        for i in range(len(self.W)):
            self.W_op2.append(self.W[i].assign(self.Wst[i]))
            self.U_op2.append(self.U[i].assign(self.Ust[i]))
            self.B_op2.append(self.B[i].assign(self.Bst[i]))
            if i == 0:
                self.sparse_retrain_grp = tf.group(self.W_op2[i], self.U_op2[i], self.B_op2[i])
            else:
                self.sparse_retrain_grp = tf.group(self.W_op2[i], self.U_op2[i], self.B_op2[i], self.sparse_retrain_grp)
    def runHardThrsd(self, sess):
        self.W_old = []
        self.U_old = []
        self.B_old = []

        self.W_new = []
        self.U_new = []
        self.B_new = []

        for i in range(len(self.W)):
            self.W_old.append(self.W[i].eval())
            self.U_old.append(self.U[i].eval())
            self.B_old.append(self.B[i].eval())

            self.W_new.append(hard_thrsd(self.W_old[i], self.sW))
            self.U_new.append(hard_thrsd(self.U_old[i], self.sU))
            self.B_new.append(hard_thrsd(self.B_old[i], self.sB))

        fd_thrsd = {}
        for i in range(len(self.W)):
            fd_thrsd[self.Wth[i]] = self.W_new[i]
            fd_thrsd[self.Uth[i]] = self.U_new[i]
            fd_thrsd[self.Bth[i]] = self.B_new[i]

        sess.run(self.hard_thrsd_grp, feed_dict=fd_thrsd)


    def runSparseTraining(self, sess):
        self.W_old = []
        self.U_old = []
        self.B_old = []

        self.W_new1 = []
        self.U_new1 = []
        self.B_new1 = []

        for i in range(len(self.W)):
            self.W_old.append(self.W[i].eval())
            self.U_old.append(self.U[i].eval())
            self.B_old.append(self.B[i].eval())

            self.W_new1.append(copy_support(self.W_new[i], self.W_old[i]))
            self.U_new1.append(copy_support(self.U_new[i], self.U_old[i]))
            self.B_new1.append(copy_support(self.B_new[i], self.B_old[i]))

        fd_st = {}
        for i in range(len(self.W)):
            fd_st[self.Wst[i]] = self.W_new1[i]
            fd_st[self.Ust[i]] = self.U_new1[i]
            fd_st[self.Bst[i]] = self.B_new1[i]

        sess.run(self.sparse_retrain_grp, feed_dict=fd_st)
    def compute(self,x):
        state=[]
        cell_state=[]
        for i in range(self.num_timesteps):
            #print(x[0])    
            if(i==0):
                I=tf.sigmoid(tf.matmul(x[i], self.W[0])+self.B[0])
                f=tf.sigmoid(tf.matmul(x[i], self.W[1])+self.B[1])
                o=tf.sigmoid(tf.matmul(x[i], self.W[2])+self.B[2])
                g=tf.tanh(tf.matmul(x[i], self.W[3])+self.B[3])
                cell_state=tf.multiply(I,g)
                state=tf.multiply(o,tf.tanh(cell_state))

            else:            
                I=tf.sigmoid(tf.matmul(x[i], self.W[0])+tf.matmul(state,self.U[0])+self.B[0])
                f=tf.sigmoid(tf.matmul(x[i], self.W[1])+tf.matmul(state,self.U[1])+self.B[1])
                o=tf.sigmoid(tf.matmul(x[i], self.W[2])+tf.matmul(state,self.U[2])+self.B[2])
                g=tf.tanh(tf.matmul(x[i], self.W[3])+tf.matmul(state,self.U[3])+self.B[3])
                cell_state=tf.multiply(f,cell_state)+tf.multiply(I,g) ##check forget bias with f+1 with cell state 
                state=tf.multiply(o,tf.tanh(cell_state))
        
        return state

def RNN(x,cell_type=None):

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    custom_output=[]
    if(cell_type=='lstm'):
        #custom_output=custom_lstm(x,timesteps,num_input,num_hidden)
        cell =lstm(timesteps,num_input,num_hidden)
        custom_output=cell.compute(x)

    elif(cell_type=='gru'):
        cell=gru(timesteps,num_input,num_hidden)
        custom_output=cell.compute(x)
    else:
        #custom_output=custom_rnn(x,timesteps,num_input,num_hidden)        
        cell=rnn(timesteps,num_input,num_hidden)
        custom_output=cell.compute(x)

    return cell, custom_output

cell_type = sys.argv[7]
lstm_cell, feats = RNN(X, cell_type)
lstm_cell.placeholder()
logits = tf.matmul(feats, weights['out']) + biases['out']
prediction = logits
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
train_op = optimizer.minimize(loss_op)
# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# W = int(os.popen("pidstat -r -p " + str(pid)).read().splitlines()[3].split()[-3])   
# print("Memory:"+str(W))

# def heron(pid,mem):
#     max_mem=-10000
#     while(True):    
#         W = int(os.popen("pidstat -r -p " + str(pid)).read().splitlines()[3].split()[-3])   
#         if(W-mem>max_mem):
#             max_mem=W-mem       
#             print("max_mem:"+str(max_mem)+" kB")

# Start training
f = open(data_dir+str(cell_type)+'_custom_imple_results.txt','a+')
f.write(str(num_input)+" "+str(timesteps)+" "+str(num_hidden)+" ")
trimlevel = 15
iht_done = 0
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(init)
    old_accu=-10000.0
    max_accu=-10000.0   
    max_train_accu=-1000.0  
    num_iters=train_feats.shape[0]/batch_size
    total_batches = num_iters * training_steps
    counter = 0
    for step in range(0, training_steps):
        loss1=0
        acc=0
        for j in range(num_iters):
            batch_x=train_feats[j*batch_size:(j+1)*batch_size]
            batch_y=train_labels[j*batch_size:(j+1)*batch_size]
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            _,temp1,temp2=sess.run([train_op,loss_op,accuracy], feed_dict={X: batch_x, Y: batch_y})
            loss1=loss1+temp1
            acc=acc+temp2
        batch_x=train_feats[num_iters*batch_size:train_feats.shape[0]]
        batch_y=train_labels[num_iters*batch_size:train_feats.shape[0]]
        batch_x = batch_x.reshape((train_feats.shape[0]-num_iters*batch_size, timesteps, num_input))
        temp1,temp2=sess.run([loss_op,accuracy], feed_dict={X: batch_x, Y: batch_y})
        print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss1/num_iters) + ", Training Accuracy= " + "{:.3f}".format(acc/num_iters))
        if((acc/num_iters)>max_train_accu):
            max_train_accu=(acc/num_iters)
        test_data = test_feats.reshape((-1, timesteps, num_input))
        test_label = test_labels 
        test_acc=sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
        print("Testing Accuracy:", str(test_acc))
        if(old_accu>acc):
            learning_rate=learning_rate*0.9
        if(test_acc>max_accu):
            max_accu=test_acc
            step_num = step
    f.write(" Test Accuracy: "+str(max_accu)+" "+str(test_acc)+" "+str(step_num))
    f.write(" Train Accuracy: "+str(max_train_accu)+" "+str(cell_type))
    all_var=sess.run(tf.trainable_variables())  
    allweights=0         
    for p in all_var:   
        allweights=allweights+np.count_nonzero(p)  
    print("Trainable_variables: "+str(allweights*8/1024)+str(" KB"))
    f.write(" Trainable_variables: "+str(allweights*8/1024)+str(" KB \n"))
    f.close()