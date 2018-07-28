import tensorflow as tf
import numpy as np

def multi_class_hinge_loss(logits, label, batch_th):
	flat_logits = tf.reshape(logits, [-1,])
	correct_id = tf.range(0, batch_th) * logits.shape[1] + label
	correct_logit = tf.gather(flat_logits, correct_id)

	max_label = tf.argmax(logits, 1)
	top2, _ = tf.nn.top_k(logits, k=2, sorted=True)

	wrong_max_logit = tf.where(tf.equal(max_label, label), top2[:,1], top2[:,0])

	return tf.reduce_mean(tf.nn.relu(1. + wrong_max_logit - correct_logit))

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
	def __init__(self, C, F, P, D, S, lW, lT, lV, lZ, 
		sW, sT, sV, sZ, lr = None, W = None, T = None, V = None, Z = None):

		self.dataDimension = F + 1
		self.projectionDimension = P
		if (C > 2):
			self.numClasses = C
		elif (C == 2):
			self.numClasses = 1

		self.treeDepth = D
		self.sigma = S #tf.Variable(S, name='sigma', dtype=tf.float32)
		self.lW = lW
		self.lV = lV
		self.lT = lT
		self.lZ = lZ
		self.sW = sW
		self.sV = sV
		self.sT = sT
		self.sZ = sZ

		self.internalNodes = 2**self.treeDepth - 1
		self.totalNodes = 2*self.internalNodes + 1

		self.W = self.initW(W)
		self.V = self.initV(V)
		self.T = self.initT(T)
		self.Z = self.initZ(Z)

		self.W_th = tf.placeholder(tf.float32, name='W_th')
		self.V_th = tf.placeholder(tf.float32, name='V_th')
		self.Z_th = tf.placeholder(tf.float32, name='Z_th')
		self.T_th = tf.placeholder(tf.float32, name='T_th')

		self.W_st = tf.placeholder(tf.float32, name='W_st')
		self.V_st = tf.placeholder(tf.float32, name='V_st')
		self.Z_st = tf.placeholder(tf.float32, name='Z_st')
		self.T_st = tf.placeholder(tf.float32, name='T_st')
		
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
		Z = tf.Variable(Z, name='Z', dtype=tf.float32)
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
		nnzZ = np.ceil(int(Z.shape[0]*Z.shape[1])*sZ)
		nnzW = np.ceil(int(W.shape[0]*W.shape[1])*sW)
		nnzV = np.ceil(int(V.shape[0]*V.shape[1])*sV)
		nnzT = np.ceil(int(T.shape[0]*T.shape[1])*sT)
		return (nnzZ+nnzT+nnzV+nnzW)*8

	def bonsaiGraph(self, X):
		X = tf.reshape(X,[-1,self.dataDimension])
		X_ = tf.divide(tf.matmul(self.Z, X, transpose_b=True), self.projectionDimension)
		# X_ = tf.nn.l2_normalize(tf.matmul(self.Z, X, transpose_b=True), 0)
		W_ = self.W[0:(self.numClasses)]
		V_ = self.V[0:(self.numClasses)]
		self.nodeProb = []
		self.nodeProb.append(1)
		score_ = self.nodeProb[0]*tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma*tf.matmul(V_, X_)))
		for i in range(1, self.totalNodes):
			W_ = self.W[i*self.numClasses:((i+1)*self.numClasses)]
			V_ = self.V[i*self.numClasses:((i+1)*self.numClasses)]
			prob = (1+((-1)**(i+1))*tf.tanh(tf.multiply(self.sigmaI, 
				tf.matmul(tf.reshape(self.T[int(np.ceil(i/2)-1)], [-1, self.projectionDimension]), X_))))
			prob = tf.divide(prob, 2)
			prob = self.nodeProb[int(np.ceil(i/2)-1)]*prob
			self.nodeProb.append(prob)
			score_ = score_ + self.nodeProb[i]*tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma*tf.matmul(V_, X_)))
			
		return score_, X_, self.T, self.W, self.V, self.Z

	def hardThrsd(self):
		self.W_op1 = self.W.assign(self.W_th)
		self.V_op1 = self.V.assign(self.V_th)
		self.T_op1 = self.T.assign(self.T_th)
		self.Z_op1 = self.Z.assign(self.Z_th)
		self.hard_thrsd_grp = tf.group(self.W_op1, self.V_op1, self.T_op1, self.Z_op1)

	def sparseTraining(self):
		self.W_op2 = self.W.assign(self.W_st)
		self.V_op2 = self.V.assign(self.V_st)
		self.Z_op2 = self.Z.assign(self.Z_st)
		self.T_op2 = self.T.assign(self.T_st)
		self.sparse_retrain_grp = tf.group(self.W_op2, self.V_op2, self.T_op2, self.Z_op2)

	def lossGraph(self):
		self.score, self.X_eval, self.T_eval, self.W_eval, self.V_eval, self.Z_eval = self.bonsaiGraph(self.x)

		if (self.numClasses > 2):
			self.margin_loss = multi_class_hinge_loss(tf.transpose(self.score), tf.argmax(self.y,1), self.batch_th)
			self.reg_loss = 0.5*(self.lZ*tf.square(tf.norm(self.Z)) + self.lW*tf.square(tf.norm(self.W)) + 
				self.lV*tf.square(tf.norm(self.V)) + self.lT*tf.square(tf.norm(self.T)))
			self.loss = self.margin_loss + self.reg_loss
		else:
			self.margin_loss = tf.reduce_mean(tf.nn.relu(1.0 - (2*self.y-1)*tf.transpose(self.score)))
			self.reg_loss = 0.5*(self.lZ*tf.square(tf.norm(self.Z)) + self.lW*tf.square(tf.norm(self.W)) + 
				self.lV*tf.square(tf.norm(self.V)) + self.lT*tf.square(tf.norm(self.T)))
			self.loss = self.margin_loss + self.reg_loss

	def trainGraph(self):
		self.train_stepW = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.W]))
		self.train_stepV = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.V]))
		self.train_stepT = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.T]))
		self.train_stepZ = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.Z]))

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




tf.set_random_seed(42)
np.random.seed(42)


inp_dims=784
n_classes=10
batch_size=100
sigma=4
depth = 3

embed_dims=10
reg_Z=0.001
reg_T=0.01
reg_W=0.01
reg_V=0.01

spar_Z=0.1
spar_W=0.5
spar_V=0.5
spar_T=0.5

learning_rate=0.01
train=np.load('/home/pranshu/Desktop/Academics/IITD/Ph.D/Research/Datasets/MNIST10/train.npy')
test=np.load('/home/pranshu/Desktop/Academics/IITD/Ph.D/Research/Datasets/MNIST10/test.npy')

train_feats=train[:,1:inp_dims+1]
train_lbl=train[:,0]
test_feats=test[:,1:inp_dims+1]
test_lbl=test[:,0]
train_lbl=train_lbl
test_lbl=test_lbl

mean=np.mean(train_feats,0)
std=np.std(train_feats,0)
std[std[:]<0.00001]=1
train_feats=(train_feats-mean)/std

test_feats=(test_feats-mean)/std

lab=train_lbl.astype('uint8')
lab=np.array(lab)-min(lab)

lab1= np.zeros((train_feats.shape[0], n_classes))
lab1[np.arange(train_feats.shape[0]), lab] = 1
train_labels=lab1
if (n_classes==2):
	train_labels = lab
	train_labels = np.reshape(train_labels, [-1, 1])

lab=test_lbl.astype('uint8')
lab=np.array(lab)-min(lab)
lab1= np.zeros((test_feats.shape[0], n_classes))
lab1[np.arange(test_feats.shape[0]), lab] = 1
test_labels=lab1
train_bias=np.ones([train_feats.shape[0],1]);
train_feats=np.append(train_feats,train_bias,axis=1)
test_bias=np.ones([test_feats.shape[0],1]);
test_feats=np.append(test_feats,test_bias,axis=1)
if (n_classes == 2):
	test_labels = lab
	test_labels = np.reshape(test_labels, [-1, 1])

batch_size = np.maximum(100, int(np.ceil(np.sqrt(train_labels.shape[0]))))

# W1 = np.load("W.npy")
# V1 = np.load("V.npy")
# T1 = np.load("T.npy")
# Z1 = np.load("Z.npy")
bonsaiObj = Bonsai(n_classes, inp_dims, embed_dims, depth, sigma, 
	reg_W, reg_T, reg_V, reg_Z, spar_W, spar_T, spar_V, spar_Z, learning_rate)




sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_variables(tf.local_variables())))
saver = tf.train.Saver()   ##for saving the model
        
total_epochs = 100
num_iters=train_feats.shape[0]/batch_size

total_batches = num_iters*total_epochs

counter = 0
if bonsaiObj.numClasses > 2:
	trimlevel = 15
else:
	trimlevel = 5
iht_done = 0


for i in range(total_epochs):

	accu = 0.0
	for j in range(num_iters):

		if ((counter == 0) or (counter == total_batches/3) or (counter == 2*total_batches/3)):
			bonsaiObj.sigmaI = 1
			iters_phase = 0

		elif (iters_phase%100 == 0):
			indices = np.random.choice(train_feats.shape[0],100)
			batch_x = train_feats[indices,:]
			batch_y = train_labels[indices,:]
			batch_y = np.reshape(batch_y, [-1, bonsaiObj.numClasses])
			_feed_dict = {bonsaiObj.x: batch_x, bonsaiObj.y: batch_y}
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
		batch_y = np.reshape(batch_y, [-1, bonsaiObj.numClasses])

		if bonsaiObj.numClasses > 2:
			_feed_dict = {bonsaiObj.x: batch_x, bonsaiObj.y: batch_y, bonsaiObj.batch_th: batch_y.shape[0]}
		else:
			_feed_dict = {bonsaiObj.x: batch_x, bonsaiObj.y: batch_y}

		_,loss1 = sess.run([bonsaiObj.train_stepW, bonsaiObj.loss], feed_dict=_feed_dict)
		_,loss1 = sess.run([bonsaiObj.train_stepV, bonsaiObj.loss], feed_dict=_feed_dict)
		_,loss1 = sess.run([bonsaiObj.train_stepT, bonsaiObj.loss], feed_dict=_feed_dict)
		_,loss1 = sess.run([bonsaiObj.train_stepZ, bonsaiObj.loss], feed_dict=_feed_dict)
		temp = bonsaiObj.accuracy.eval(feed_dict=_feed_dict)
		accu = temp+accu

		if (counter >= (total_batches/3) and (counter < 2*total_batches/3) and counter%trimlevel == 0):
			W_old = bonsaiObj.W_eval.eval()
			V_old = bonsaiObj.V_eval.eval()
			Z_old = bonsaiObj.Z_eval.eval()
			T_old = bonsaiObj.T_eval.eval()

			W_new = hard_thrsd(W_old, bonsaiObj.sW)
			V_new = hard_thrsd(V_old, bonsaiObj.sV)
			Z_new = hard_thrsd(Z_old, bonsaiObj.sZ)
			T_new = hard_thrsd(T_old, bonsaiObj.sT)

			if counter%num_iters == 0:
				print("IHT", np.count_nonzero(W_new), np.count_nonzero(V_new), np.count_nonzero(Z_new), np.count_nonzero(T_new))

			fd_thrsd = {bonsaiObj.W_th:W_new, bonsaiObj.V_th:V_new, bonsaiObj.Z_th:Z_new, bonsaiObj.T_th:T_new}
			sess.run(bonsaiObj.hard_thrsd_grp, feed_dict=fd_thrsd)

			iht_done = 1
		elif ((iht_done == 1 and counter >= (total_batches/3) and (counter < 2*total_batches/3) and counter%trimlevel != 0) or (counter >= 2*total_batches/3)):
			W_old = bonsaiObj.W_eval.eval()
			V_old = bonsaiObj.V_eval.eval()
			Z_old = bonsaiObj.Z_eval.eval()
			T_old = bonsaiObj.T_eval.eval()

			W_new1 = copy_support(W_new, W_old)
			V_new1 = copy_support(V_new, V_old)
			Z_new1 = copy_support(Z_new, Z_old)
			T_new1 = copy_support(T_new, T_old)

			if counter%num_iters == 0:
				print("ST", np.count_nonzero(W_new1), np.count_nonzero(V_new1), np.count_nonzero(Z_new1), np.count_nonzero(T_new1))
				print(8.0*(np.count_nonzero(W_new) + np.count_nonzero(V_new) + np.count_nonzero(Z_new) + np.count_nonzero(T_new))/1024.0)
			fd_st = {bonsaiObj.W_st:W_new1, bonsaiObj.V_st:V_new1, bonsaiObj.Z_st:Z_new1, bonsaiObj.T_st:T_new1}
			sess.run(bonsaiObj.sparse_retrain_grp, feed_dict=fd_st)

		counter = counter + 1

	print("Train accuracy "+str(accu/num_iters)) 
	bonsaiObj.analyse()
	if bonsaiObj.numClasses > 2:
		_feed_dict={bonsaiObj.x: test_feats, bonsaiObj.y: test_labels, bonsaiObj.batch_th: test_labels.shape[0]}
	else:
		_feed_dict={bonsaiObj.x: test_feats, bonsaiObj.y: test_labels}

	old = bonsaiObj.sigmaI
	bonsaiObj.sigmaI = 1e9

	print("Test accuracy %g"%bonsaiObj.accuracy.eval(feed_dict=_feed_dict))
	if bonsaiObj.numClasses > 2:
		_feed_dict={bonsaiObj.x: train_feats, bonsaiObj.y: train_labels, bonsaiObj.batch_th: train_labels.shape[0]}
	else:
		_feed_dict={bonsaiObj.x: train_feats, bonsaiObj.y: train_labels}

	loss_new = bonsaiObj.loss.eval(feed_dict=_feed_dict)
	reg_loss_new = bonsaiObj.reg_loss.eval(feed_dict=_feed_dict)
	print("Loss %g"%loss_new)
	print("Reg Loss %g"%reg_loss_new)
	print("Train accuracy %g"%bonsaiObj.accuracy.eval(feed_dict=_feed_dict))

	bonsaiObj.sigmaI = old
	print(old)

	print("\n Epoch Number: "+str(i+1))   

np.save("W.npy", bonsaiObj.W_eval.eval())
np.save("V.npy", bonsaiObj.V_eval.eval())
np.save("Z.npy", bonsaiObj.Z_eval.eval())
np.save("T.npy", bonsaiObj.T_eval.eval())
