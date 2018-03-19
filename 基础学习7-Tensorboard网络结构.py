import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/Users/zero/Downloads/',one_hot=True)

#每个批次的大小
batch_size =100
n_batch =mnist.train.num_examples//batch_size

#命名空间
with tf.name_scope('input'):
	#定义两个placeholder
	y = tf.placeholder(tf.float32,[None,10],name='x-input')
	x = tf.placeholder(tf.float32,[None,784],name='y-input')
#keep_prob=tf.placeholder(tf.float32)

with tf.name_scope('layer'):
	#创建一个简单的神经网络  s
	with tf.name_scope('wights'):
		W = tf.Variable(tf.zeros([784,10]))
	with tf.name_scope('biases'):
		b = tf.Variable(tf.zeros([10]))
	with tf.name_scope('wx_plus_b'):
		wx_plus_b = tf.matmul(x,W)+b 
	with tf.name_scope('softmax'):	
		prediction = tf.nn.softmax(wx_plus_b) #损失函数

#二次代价函数
with tf.name_scope('loss'):
	loss =tf.reduce_mean(tf.square(y-prediction)) 
#对数
#loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)) #交叉熵
#使用梯度下降法
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
	#argmax返回一维张量中最大的值所在的位置
	with tf.name_scope('accuracy'):
		#求准确率
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter('logs/',sess.graph)
	for epoch in range(1):
		for batch in range(n_batch):
			batch_xs,batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

		acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
		#train_acc= sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
		print("Iter"+ str(epoch)+ ", Testing Accuracy " + str(acc))
