import tensorflow as tf
import numpy as np

tf.set_random_seed(42)
np.random.seed(42)

def forward_iter(data, labels, sigI, index, code):
    batchx = data[index];  batchy = labels[index]; 
    if(code): sess.run(train_op, feed_dict={X: batchx, Y:batchy, sigmaI: sigI})
    else: return(sess.run(accuracy, feed_dict={X: batchx, Y: batchy, sigmaI: sigI}))

'''
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
'''

train_data = []
train_labels = []
X = np.load("train.npy")
Y = np.load("test.npy")
nc = int(np.max(X[:,0])) - int(np.min(X[:,0])) + 1
X = np.concatenate((X,np.ones((X.__len__(),1))),axis=1)
Y = np.concatenate((Y,np.ones((Y.__len__(),1))),axis=1)
train_data = X[:,1:]
train_labels = np.zeros((X.shape[0], nc)); train_labels[np.arange(X.shape[0]),np.array((X[:,0]-np.min(X[:,0])).tolist(),dtype=int)] = 1;
test_data = Y[:,1:]
test_labels = np.zeros((Y.shape[0], nc)); test_labels[np.arange(Y.shape[0]),np.array((Y[:,0]-np.min(Y[:,0])).tolist(),dtype=int)] = 1;

mean=np.mean(train_data,0)
std=np.std(train_data,0)
std[std[:]<0.00001]=1
train_data=(train_data-mean)/std
test_data=(test_data-mean)/std

h = 3
pd = 10
lr = 0.01
nf = int(train_data.shape[1])

lT = 1e-2
lW = 1e-2
lV = 1e-2
lZ = 1e-3
sT = 0.5
sW = 0.5
sV = 0.5
sZ = 0.1
batch_size = 100
sigma = 4

class Bonsai:
    def __init__(self,h,pd,nf,nc,lT,lW,lV,lZ,sT,sW,sV,sZ,sigma):
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

        self.sigma = sigma

        self.Z = tf.Variable(tf.random_normal([self.pd, self.nf]), name='Z', dtype=tf.float32)

        #if(self.int_n > 0):
        self.T = tf.Variable(tf.random_normal([self.int_n, self.pd]), name='T', dtype=tf.float32)

        self.V = tf.Variable(tf.random_normal([self.tot_n, self.nc, self.pd]), name='V', dtype=tf.float32)
        self.W = tf.Variable(tf.random_normal([self.tot_n, self.nc, self.pd]), name='W', dtype=tf.float32)
        
    def forward(self, X, sigmaI):
        X_ = tf.divide(tf.matmul(self.Z, X, transpose_b=True), self.pd)
       	W_ = self.W[0]; V_ = self.V[0];
        self.nodeProb = []
        self.nodeProb.append(1)
        score_ = self.nodeProb[0]*tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma*tf.matmul(V_, X_)))
        for i in range(1, self.tot_n):
            W_ = self.W[i]; V_ = self.V[i];
            prob = (1+((-1)**(i+1))*tf.tanh(tf.multiply(sigmaI,tf.matmul(tf.reshape(self.T[int(np.ceil(i/2)-1)], [-1, self.pd]), X_))))
            prob = tf.divide(prob, 2)
            prob = self.nodeProb[int(np.ceil(i/2)-1)]*prob
            self.nodeProb.append(prob)
            score_ = score_ + self.nodeProb[i]*tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma*tf.matmul(V_, X_)))
        return tf.transpose(score_)

    def multi_class_loss(self, outputs, labels):
        multi_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels))
        reg_loss = 0.5*(self.lZ*tf.square(tf.norm(self.Z)) + self.lW*tf.square(tf.norm(self.W)) +  self.lV*tf.square(tf.norm(self.V)) + self.lT*tf.square(tf.norm(self.T)))
        total_loss = multi_loss + reg_loss
        return total_loss

bonsai = Bonsai(h,pd,nf,nc,lT,lW,lV,lZ,sT,sW,sV,sZ,sigma)

X = tf.placeholder("float", [None, nf])
Y = tf.placeholder("float", [None, nc])
sigmaI = tf.placeholder("float", None)

logits = bonsai.forward(X, sigmaI)

prediction = tf.nn.softmax(logits)
pred_labels = tf.argmax(prediction, 1)

correct_pred = tf.equal(pred_labels, tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss_op = bonsai.multi_class_loss(logits, Y) 
optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9,use_nesterov=True) 
train_op = optimizer.minimize(loss_op)

total_epochs = 50
#total_batches = num_iters * total_epochs
#counter = 0
'''
if bonsai.nc > 2:
	trimlevel = 15
else:
	trimlevel = 5
'''
#iht_done = 0
sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_variables(tf.local_variables()))) 

for i in range(total_epochs):
    sigI = 1
    num_iters = int(train_data.shape[0]/batch_size)
    [forward_iter(train_data,train_labels,sigI,slice(j*batch_size,(j+1)*batch_size),True) for j in range(num_iters)]
    forward_iter(train_data,train_labels,sigI,slice(num_iters*batch_size,train_data.shape[0]),True)
    acc = forward_iter(test_data,test_labels,1e9,slice(0,test_data.__len__()),False)
    print(acc)
    sigI = sigI * 1.75
