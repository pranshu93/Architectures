from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import timeit
import os
from thread import start_new_thread
import sys
tf.set_random_seed(42)
np.random.seed(42)

with tf.device('/device:CPU:0'):

	def multi_class_hinge_loss(logits, label, batch_th):
	    flat_logits = tf.reshape(logits, [-1,])
	    correct_id = tf.range(0, batch_th) * logits.shape[1] + label
	    correct_logit = tf.gather(flat_logits, correct_id)

	    max_label = tf.argmax(logits, 1)
	    top2, _ = tf.nn.top_k(logits, k=2, sorted=True)

	    wrong_max_logit = tf.where(tf.equal(max_label, label), top2[:,1], top2[:,0])

	    return tf.reduce_mean(tf.nn.relu(1. + wrong_max_logit - correct_logit))

	def cross_entropy_loss(logit,label):
	    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label))  

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


	class Bonsai:
	    def __init__(self, C, F, P, D, S, lW, lT, lV, 
	        sW, sT, sV, lr = None, feats_ = None, W = None, T = None, V = None):

	        self.dataDimension = F + 1
	        self.projectionDimension = P
	        if (C > 2):
	            self.numClasses = C
	        elif (C == 2):
	            self.numClasses = 1

	        self.treeDepth = D
	        self.sigma = S
	        self.lW = lW
	        self.lV = lV
	        self.lT = lT
	        #self.lZ = lZ
	        self.sW = sW
	        self.sV = sV
	        self.sT = sT
	        #self.sZ = sZ

	        self.internalNodes = 2**self.treeDepth - 1
	        self.totalNodes = 2*self.internalNodes + 1

	        self.W = self.initW(W)
	        self.V = self.initV(V)
	        self.T = self.initT(T)
	        #self.Z = self.initZ(Z)

	        self.W_th = tf.placeholder(tf.float32, name='W_th')
	        self.V_th = tf.placeholder(tf.float32, name='V_th')
	        #self.Z_th = tf.placeholder(tf.float32, name='Z_th')
	        self.T_th = tf.placeholder(tf.float32, name='T_th')

	        self.W_st = tf.placeholder(tf.float32, name='W_st')
	        self.V_st = tf.placeholder(tf.float32, name='V_st')
	        #self.Z_st = tf.placeholder(tf.float32, name='Z_st')
	        self.T_st = tf.placeholder(tf.float32, name='T_st')
	        
	        if feats_ is not None:
	            self.x = feats_
	        else:
	            self.x = tf.placeholder("float", [None, self.dataDimension])
	        self.y = tf.placeholder("float", [None, self.numClasses])
	        self.batch_th = tf.placeholder(tf.int64, name='batch_th')

	        self.sigmaI = 1.0
	        if lr is not None:
	            self.learning_rate = lr
	        else:
	            self.learning_rate = 0.01

	        self.hardThrsd()
	        self.sparseTraining()
	        self.lossGraph()
	        self.trainGraph()
	        self.accuracyGraph()

	    def initZ(self, Z):
	        if Z is None:
	            Z = tf.random_normal([self.projectionDimension, self.dataDimension])
	        Z_ini=np.identity(self.dataDimension)
	        Z = tf.Variable(Z_ini, name='Z', dtype=tf.float32)
	        return Z

	    def initW(self, W):
	        if W is None:
	            W = tf.random_normal([self.numClasses*self.totalNodes, self.projectionDimension])
	        W = tf.Variable(W, name='W', dtype=tf.float32)
	        return W

	    def initV(self, V):
	        if V is None:
	            V = tf.random_normal([self.numClasses*self.totalNodes, self.projectionDimension])
	        V = tf.Variable(V, name='V', dtype=tf.float32)
	        return V

	    def initT(self, T):
	        if T is None:
	            T = tf.random_normal([self.internalNodes, self.projectionDimension])
	        T = tf.Variable(T, name='T', dtype=tf.float32)
	        return T

	    def getModelSize(self): 
	        #nnzZ = np.ceil(int(Z.shape[0]*Z.shape[1])*sZ)
	        nnzW = np.ceil(int(W.shape[0]*W.shape[1])*sW)
	        nnzV = np.ceil(int(V.shape[0]*V.shape[1])*sV)
	        nnzT = np.ceil(int(T.shape[0]*T.shape[1])*sT)
	        return (nnzT+nnzV+nnzW)*8

	    def bonsaiGraph(self, X):
	        X = tf.reshape(X,[-1,self.dataDimension])
	    #print(X)        
	    #X_ = tf.divide(tf.matmul(self.Z, X, transpose_b=True), self.projectionDimension)
	    #X_=tf.transpose(tf.divide(X,self.projectionDimension))
	        X_=tf.transpose(X)        
	        W_ = self.W[0:(self.numClasses)]
	        V_ = self.V[0:(self.numClasses)]
	        self.nodeProb = []
	        self.nodeProb.append(1)
	    #biases = {
	    #    'out': tf.Variable(tf.random_normal([13]))
	    #}
	    #print("yoyo")
	    #print(X_)
	        #score_ = self.nodeProb[0]*tf.matmul(W_, X_)#,biases['out'])
	    #score_=tf.transpose(score_)+biases['']
	    #print(score_)
	        score_ = self.nodeProb[0]*tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma*tf.matmul(V_, X_)))
	        print("Total nodes: "+str(self.totalNodes))
	        for i in range(1, self.totalNodes):
	            W_ = self.W[i*self.numClasses:((i+1)*self.numClasses)]
	            V_ = self.V[i*self.numClasses:((i+1)*self.numClasses)]
	            prob = (1+((-1)**(i+1))*tf.tanh(tf.multiply(self.sigmaI, 
	                tf.matmul(tf.reshape(self.T[int(np.ceil(i/2)-1)], [-1, self.projectionDimension]), X_))))
	            prob = tf.divide(prob, 2)
	            prob = self.nodeProb[int(np.ceil(i/2)-1)]*prob
	            self.nodeProb.append(prob)
	            score_ = score_ + self.nodeProb[i]*tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma*tf.matmul(V_, X_)))
	            
	        return score_, X_, self.T, self.W, self.V

	    def hardThrsd(self):
	        self.W_op1 = self.W.assign(self.W_th)
	        self.V_op1 = self.V.assign(self.V_th)
	        self.T_op1 = self.T.assign(self.T_th)
	        #self.Z_op1 = self.Z.assign(self.Z_th)
	        #self.hard_thrsd_grp = tf.group(self.W_op1, self.V_op1, self.T_op1, self.Z_op1)
	        self.hard_thrsd_grp = tf.group(self.W_op1, self.V_op1, self.T_op1)
	 
	    def sparseTraining(self):
	        self.W_op2 = self.W.assign(self.W_st)
	        self.V_op2 = self.V.assign(self.V_st)
	        #self.Z_op2 = self.Z.assign(self.Z_st)
	        self.T_op2 = self.T.assign(self.T_st)
	        #self.sparse_retrain_grp = tf.group(self.W_op2, self.V_op2, self.T_op2, self.Z_op2)
	        self.sparse_retrain_grp = tf.group(self.W_op2, self.V_op2, self.T_op2)

	    def lossGraph(self):
	        self.score, self.X_eval, self.T_eval, self.W_eval, self.V_eval= self.bonsaiGraph(self.x)

	        if (self.numClasses > 2):
	            #self.margin_loss = multi_class_hinge_loss(tf.transpose(self.score), tf.argmax(self.y,1), self.batch_th)
	        #print(tf.transpose(self.score))
	        #print(self.y)  
	            self.margin_loss=cross_entropy_loss(tf.transpose(self.score),self.y)
	            self.reg_loss = 0.5*(self.lW*tf.square(tf.norm(self.W)) + 
	                self.lV*tf.square(tf.norm(self.V)) + self.lT*tf.square(tf.norm(self.T)))
	            self.loss = self.margin_loss + self.reg_loss
	        else:
	            self.margin_loss = tf.reduce_mean(tf.nn.relu(1.0 - (2*self.y-1)*tf.transpose(self.score)))
	            self.reg_loss = 0.5*(self.lW*tf.square(tf.norm(self.W)) + 
	                self.lV*tf.square(tf.norm(self.V)) + self.lT*tf.square(tf.norm(self.T)))
	            self.loss = self.margin_loss + self.reg_loss

	    def trainGraph(self):
	        #w=tf.trainable
	        rnn_weights=[]
	        all_vars=[]
	        #for p in tf.trainable_variables():
	        #    print(p.name)  
	        #    if 'Z:0' not in p.name:
	        #        all_vars.append(p)    
	        #    if 'rnn' in p.name:
	        #        rnn_weights.append(p)
	        #        print(p.name)                        
	        #self.train_stepRNN = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[rnn_weights]))
	        self.train_stepW = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.W]))
	        self.train_stepV = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.V]))
	        self.train_stepT = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.T]))
	        #self.train_stepZ = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.Z]))
	        #self.train_all= tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)
	        self.train_all = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(self.loss)
	    #print(all_vars)
	        #self.train_all=(tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss))
	    def accuracyGraph(self):
	        if (self.numClasses > 2):
	            correct_prediction = tf.equal(tf.argmax(tf.transpose(self.score),1), tf.argmax(self.y,1))
	            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	        else:
	            y_ = self.y*2-1
	            correct_prediction = tf.multiply(tf.transpose(self.score), y_)
	            correct_prediction = tf.nn.relu(correct_prediction)
	            correct_prediction = tf.ceil(tf.tanh(correct_prediction))
	            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	    def analyse(self):
	        _feed_dict={self.x: train_feats, self.y: train_labels}  
	        x_cap_eval = self.X_eval.eval(feed_dict=_feed_dict)
	        tt1 = self.T_eval.eval()
	        prob = []
	        for i in range(0, self.internalNodes):
	            prob.append(np.dot(tt1[i], x_cap_eval))
	        prob = np.array(prob)
	        nodes = np.zeros(self.internalNodes + 1)
	        for i in range(x_cap_eval.shape[1]):
	            j = 0
	            while j < self.internalNodes:
	                if (prob[j][i] > 0):
	                    if (2*j+1 < self.internalNodes):
	                        j = 2*j+1
	                    else:
	                        j = 2*j+1
	                        nodes[j - self.internalNodes] = nodes[j - self.internalNodes]+1
	                else:
	                    if (2*j+2 < self.internalNodes):
	                        j = 2*j+2
	                    else:
	                        j = 2*j+2
	                        nodes[j - self.internalNodes] = nodes[j - self.internalNodes]+1
	        for i in range(0, self.internalNodes + 1):
	            print(i, nodes[i])

	pid = os.getpid()

	data_dir=sys.argv[1]
	inp_dims=int(sys.argv[2])
	num_classes = int(sys.argv[3])
	#embed_dims = 3
	batch_size=100
	depth = 0

	embed_dims=int(sys.argv[6])
	#reg_Z=0.00
	reg_T=float(sys.argv[8])  #0.0000
	reg_W=float(sys.argv[9])
	reg_V=float(sys.argv[10])

	#spar_Z=1.0
	spar_T=float(sys.argv[11])
	spar_W=float(sys.argv[12])
	spar_V=float(sys.argv[13])

	sigma=float(sys.argv[14])
	cell_type=sys.argv[7]

	learning_rate=float(sys.argv[15])
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
	lab=np.array(lab)-min(lab)
	lab1= np.zeros((train_feats.shape[0], num_classes))
	lab1[np.arange(train_feats.shape[0]), lab] = 1
	train_labels=lab1
	if (num_classes==2):
	    train_labels = lab
	    train_labels = np.reshape(train_labels, [-1, 1])

	lab=test_lbl.astype('uint8')
	lab=np.array(lab)-min(lab)
	lab1= np.zeros((test_feats.shape[0], num_classes))
	lab1[np.arange(test_feats.shape[0]), lab] = 1
	test_labels=lab1
	if (num_classes == 2):
	    test_labels = lab
	    test_labels = np.reshape(test_labels, [-1, 1])

	# # Training Parameters
	# learning_rate = 0.01
	# training_steps = 1000
	# batch_size = 64
	# display_step = 200

	# Network Parameters
	num_input = int(sys.argv[4])#512 # MNIST data input (img shape: 28*28)
	timesteps =  int(sys.argv[5])#16# timesteps
	num_hidden =int(sys.argv[6])#16#hidden layer num of features

	#W = int(os.popen("pidstat -r -p " + str(pid)).read().splitlines()[3].split()[-3])  
	#print("Memory:"+str(W))

	# tf Graph input
	X_inp = tf.placeholder("float", [None, timesteps, num_input])
	# Y_inp = tf.placeholder("float", [None, num_classes])
	# lr = tf.placeholder("float")
	batch_size = 100#np.maximum(100, int(np.ceil(np.sqrt(train_labels.shape[0]))))
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
	            #print(x[0])    
	            if(i==0):
	                state=tf.tanh(tf.matmul(x[i], self.W[0])+self.B)   ##no nonlinearity better results 
	            else:            
	                state=tf.tanh(tf.matmul(x[i], self.W[0])+tf.matmul(state,self.U[0])+self.B[0])
	                
	        return state

	class gru:
	    def __init__(self,num_timesteps, dataDimensions, projectionDimensions):

	        self.dataDimensions=dataDimensions
	        self.num_timesteps=num_timesteps
	        self.projectionDimensions=projectionDimensions
	        self.W = []
	        self.U = []
	        self.B = []
	        self.sW = 0.5
	        self.sU = 0.5
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
	        custom_output=rnn_cell.compute(x)

	    bias = tf.reshape(tf.reduce_sum(custom_output,1)+1, [-1, 1])
	    return cell, tf.concat([custom_output, bias], 1)

	lstm_cell, feats = RNN(X_inp,cell_type)
	print(feats)

	bonsaiObj = Bonsai(num_classes, num_hidden, embed_dims+1, depth, sigma, 
	    reg_W, reg_T, reg_V, spar_W, spar_T, spar_V, learning_rate, feats)
	lstm_cell.placeholder()
	lstm_spar_W = float(sys.argv[16])
	lstm_spar_U = float(sys.argv[17])
	lstm_spar_B = float(sys.argv[18])
	print(lstm_spar_B)
	print(lstm_spar_U)
	print(lstm_spar_W)

	# config = tf.ConfigProto(
	#         device_count = {'CPU': 0}
	#     )
	sess = tf.InteractiveSession()
	sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_variables(tf.local_variables())))
	saver = tf.train.Saver()   ##for saving the model



	total_epochs = int(sys.argv[19])
	num_iters=train_feats.shape[0]/batch_size

	total_batches = num_iters*total_epochs

	counter = 0
	if bonsaiObj.numClasses > 2:
	    trimlevel = 15
	else:
	    trimlevel = 5
	iht_done = 0

	f = open(data_dir+'GRUBonsai_results_custom.txt','a+')
	f.write(str(embed_dims)+" "+str(timesteps)+" "+str(reg_T)+" "+str(reg_W)+" "+str(reg_V)+" "+str(spar_T)+" "+str(spar_W)+" "+str(spar_V)+" "+str(sigma)+" "+str(cell_type)+" ")
	max_accu=-1000
	test_accu=-1000
	batch_x_test = test_feats.reshape((test_feats.shape[0], timesteps, num_input))
	batch_x_train = train_feats.reshape((train_feats.shape[0], timesteps, num_input))
	best_epoch_num=-1
	for i in range(total_epochs):

	    accu = 0.0
	    for j in range(num_iters+1):
	        tf.set_random_seed(42)
	        
	        if ((counter == 0) or (counter == int(total_batches/3)) or (counter == int(2*total_batches/3))):
	            bonsaiObj.sigmaI = 1
	            iters_phase = 0

	        elif (iters_phase%100 == 0):
	            indices = np.random.choice(train_feats.shape[0],100)
	            batch_x = train_feats[indices,:]
	            batch_x = batch_x.reshape((100, timesteps, num_input))
	            batch_y = train_labels[indices,:]
	            batch_y = np.reshape(batch_y, [-1, bonsaiObj.numClasses])
	            _feed_dict = {X_inp: batch_x, bonsaiObj.y: batch_y}
	            x_cap_eval = bonsaiObj.X_eval.eval(feed_dict=_feed_dict)
	            T_eval = bonsaiObj.T_eval.eval()
	            sum_tr = 0.0
	            for k in range(0, bonsaiObj.internalNodes):
	                sum_tr = sum_tr + (np.sum(np.abs(np.dot(T_eval[k], x_cap_eval))))

	            if(bonsaiObj.internalNodes > 0):
	                sum_tr = sum_tr/(100*bonsaiObj.internalNodes)
	                sum_tr = 0.1/sum_tr
	            else:
	                sum_tr = 0.1
	            sum_tr = min(1000,sum_tr*(2**(float(iters_phase)/(float(total_batches)/30.0))))

	            bonsaiObj.sigmaI = sum_tr
	        
	        iters_phase = iters_phase + 1
	        batch_x = train_feats[j*batch_size:(j+1)*batch_size]
	        batch_y = train_labels[j*batch_size:(j+1)*batch_size]
	        batch_x = batch_x.reshape((batch_x.shape[0], timesteps, num_input))
	        batch_y = np.reshape(batch_y, [-1, bonsaiObj.numClasses])
	    
	        if bonsaiObj.numClasses > 2:
	            _feed_dict = {X_inp: batch_x, bonsaiObj.y: batch_y, bonsaiObj.batch_th: batch_y.shape[0]}
	        else:
	            _feed_dict = {X_inp: batch_x, bonsaiObj.y: batch_y}

	        #_,loss1 = sess.run([bonsaiObj.train_stepW, bonsaiObj.loss], feed_dict=_feed_dict)
	        #_,loss1 = sess.run([bonsaiObj.train_stepV, bonsaiObj.loss], feed_dict=_feed_dict)
	        #_,loss1 = sess.run([bonsaiObj.train_stepT, bonsaiObj.loss], feed_dict=_feed_dict)
	        #_,loss1 = sess.run([bonsaiObj.train_stepZ, bonsaiObj.loss], feed_dict=_feed_dict)
	        #_,loss1 = sess.run([bonsaiObj.train_stepRNN, bonsaiObj.loss], feed_dict=_feed_dict)
	        _,loss1 = sess.run([bonsaiObj.train_all, bonsaiObj.loss], feed_dict=_feed_dict)
	        temp = bonsaiObj.accuracy.eval(feed_dict=_feed_dict)
	        accu = temp+accu

	        if (counter >= (total_batches/3) and (counter < 2*total_batches/3) and counter%trimlevel == 0):
	            W_old = bonsaiObj.W_eval.eval()
	            V_old = bonsaiObj.V_eval.eval()
	            #Z_old = bonsaiObj.Z_eval.eval()
	            T_old = bonsaiObj.T_eval.eval()

	            W_new = hard_thrsd(W_old, bonsaiObj.sW)
	            V_new = hard_thrsd(V_old, bonsaiObj.sV)
	            #Z_new = hard_thrsd(Z_old, bonsaiObj.sZ)
	            T_new = hard_thrsd(T_old, bonsaiObj.sT)
	            lstm_cell.sW = lstm_spar_W
	            lstm_cell.sU = lstm_spar_U
	            lstm_cell.sB = lstm_spar_B

	            if counter%num_iters == 0:
	                print("IHT", np.count_nonzero(W_new), np.count_nonzero(V_new), np.count_nonzero(T_new))

	            fd_thrsd = {bonsaiObj.W_th:W_new, bonsaiObj.V_th:V_new, bonsaiObj.T_th:T_new}
	            sess.run(bonsaiObj.hard_thrsd_grp, feed_dict=fd_thrsd)
	            lstm_cell.runHardThrsd(sess)

	            iht_done = 1
	        elif ((iht_done == 1 and counter >= (total_batches/3) and (counter < 2*total_batches/3) and counter%trimlevel != 0) or (counter >= 2*total_batches/3)):
	            W_old = bonsaiObj.W_eval.eval()
	            V_old = bonsaiObj.V_eval.eval()
	            #Z_old = bonsaiObj.Z_eval.eval()
	            T_old = bonsaiObj.T_eval.eval()

	            W_new1 = copy_support(W_new, W_old)
	            V_new1 = copy_support(V_new, V_old)
	            #Z_new1 = copy_support(Z_new, Z_old)
	            T_new1 = copy_support(T_new, T_old)

	            if counter%num_iters == 0:
	                print("ST", np.count_nonzero(W_new1), np.count_nonzero(V_new1), np.count_nonzero(T_new1))
	                print(8.0*(np.count_nonzero(W_new) + np.count_nonzero(V_new) + np.count_nonzero(T_new))/1024.0)
	            fd_st = {bonsaiObj.W_st:W_new1, bonsaiObj.V_st:V_new1, bonsaiObj.T_st:T_new1}
	            sess.run(bonsaiObj.sparse_retrain_grp, feed_dict=fd_st)
	            lstm_cell.runSparseTraining(sess)

	        counter = counter + 1

	    print("Train accuracy "+str(accu/num_iters)) 
	    # if accu < old_accu:
	    #     learning_rate = learning_rate*0.9
	    # else:
	    #     old_accu = accu
	    # bonsaiObj.analyse()

	    if bonsaiObj.numClasses > 2:
	        batch_x = batch_x_test#test_feats.reshape((test_feats.shape[0], timesteps, num_input))
	        _feed_dict={X_inp: batch_x, bonsaiObj.y: test_labels, bonsaiObj.batch_th: test_labels.shape[0]}
	    else:
	        batch_x = batch_x_test#test_feats.reshape((test_feats.shape[0], timesteps, num_input))
	        _feed_dict={X_inp: batch_x, bonsaiObj.y: test_labels}

	    old = bonsaiObj.sigmaI
	    bonsaiObj.sigmaI = 1e9
	    test_accu=bonsaiObj.accuracy.eval(feed_dict=_feed_dict) 
	    print("Test accuracy %g"%test_accu)
	    if(max_accu<test_accu):
	        max_accu=test_accu
	    best_epoch_num=i        
	    if bonsaiObj.numClasses > 2:
	        batch_x = batch_x_train#train_feats.reshape((train_feats.shape[0], timesteps, num_input))
	        _feed_dict={X_inp: batch_x, bonsaiObj.y: train_labels, bonsaiObj.batch_th: train_labels.shape[0]}
	    else:
	        batch_x = batch_x_train#train_feats.reshape((train_feats.shape[0], timesteps, num_input))
	        _feed_dict={X_inp: batch_x, bonsaiObj.y: train_labels}

	    #loss_new = bonsaiObj.loss.eval(feed_dict=_feed_dict)
	    #reg_loss_new = bonsaiObj.reg_loss.eval(feed_dict=_feed_dict)
	    loss_new,reg_loss_new,accuracy_n=sess.run([bonsaiObj.loss,bonsaiObj.reg_loss,bonsaiObj.accuracy],feed_dict=_feed_dict)  
	    #print(sess.run(B))
	    print("Loss %g"%loss_new)
	    print("Reg Loss %g"%reg_loss_new)
	    print("Train accuracy %g"%accuracy_n)

	    bonsaiObj.sigmaI = old
	    print(old)

	    print("\n Epoch Number: "+str(i+1))   

	np.save("W.npy", bonsaiObj.W_eval.eval())
	np.save("V.npy", bonsaiObj.V_eval.eval())
	np.save("T.npy", bonsaiObj.T_eval.eval())
	all_var=sess.run(tf.trainable_variables())  
	allweights=0         
	for p in all_var:   
	    allweights=allweights+np.count_nonzero(p)      
	print("Trainable_variables: "+str(allweights*8/1024)+str(" KB"))
	f.write("Trainable_variables: "+str(allweights*8/1024)+str(" KB "))
	f.write("Test_Acc: "+str(max_accu)+" "+str(test_accu)+" "+str(best_epoch_num)+'\n')
