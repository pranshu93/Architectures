from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

tf.set_random_seed(42)
np.random.seed(42)

def forward_iter(data, labels, kp, index, code):
    batchx = data[index];  batchy = labels[index]; 
    if(code): sess.run(train_op, feed_dict={X: batchx, Y:batchy, keep_prob: kp})
    else: return(sess.run(accuracy, feed_dict={X: batchx, Y: batchy, keep_prob: kp}))

train_data = []
train_labels = []
X = np.load("/home/pranshu/Desktop/Academics/IITD/Ph.D/Research/Datasets/MNIST10/train.npy")
Y = np.load("/home/pranshu/Desktop/Academics/IITD/Ph.D/Research/Datasets/MNIST10/test.npy")
nc = int(np.max(X[:,0])) - int(np.min(X[:,0])) + 1
#X = np.concatenate((X,np.ones((X.__len__(),1))),axis=1)
#Y = np.concatenate((Y,np.ones((Y.__len__(),1))),axis=1)
train_data = X[:,1:]
train_labels = np.zeros((X.shape[0], nc)); train_labels[np.arange(X.shape[0]),np.array((X[:,0]-np.min(X[:,0])).tolist(),dtype=int)] = 1;
test_data = Y[:,1:]
test_labels = np.zeros((Y.shape[0], nc)); test_labels[np.arange(Y.shape[0]),np.array((Y[:,0]-np.min(Y[:,0])).tolist(),dtype=int)] = 1;

mean=np.mean(train_data,0)
std=np.std(train_data,0)
std[std[:]<0.00001]=1
train_data=(train_data-mean)/std
test_data=(test_data-mean)/std

nf = int(train_data.shape[1])
lr = 0.01
batch_size = 128
total_epochs = 10

X = tf.placeholder(tf.float32, [None, nf])
Y = tf.placeholder(tf.float32, [None, nc])
keep_prob = tf.placeholder(tf.float32) 

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, nc]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([nc]))
}

logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9,use_nesterov=True) 
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_variables(tf.local_variables()))) 

for i in range(total_epochs):
    sum_pred = 0;
    num_iters = int(train_data.shape[0]/batch_size)
    [forward_iter(train_data,train_labels,0.75,slice(j*batch_size,(j+1)*batch_size),True) for j in range(50)]
    #forward_iter(train_data,train_labels,1,slice(num_iters*batch_size,train_data.shape[0]),True)
    num_iters = int(test_data.shape[0]/batch_size)
    for j in range(50): sum_pred += forward_iter(test_data,test_labels,1,slice(j*batch_size,(j+1)*batch_size),False)
    #sum_pred += forward_iter(test_data,test_labels,1,slice(num_iters*batch_size,test_data.shape[0]),False)
    print(float(sum_pred)/(50*batch_size))
    #print(float(sum_pred)/test_data.shape[0])
