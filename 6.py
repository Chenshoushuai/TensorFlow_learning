'''x = tf.Variable([1,2])
a = tf.constant([3,3])
sub = tf.subtract(x,a)
add = tf.add(x,sub)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print(sess.run(sub))
	print(sess.run(add))'''

'''state = tf.Variable(0,name='counter')
new_value = tf.add(state,1)
update = tf.assign(state,new_value)#赋值

init =tf.global_variables_initializer() #初始化全局变量

with tf.Session() as sess:
	sess.run(init)
	print(sess.run(state))
	for _ in range(5):
		sess.run(update)
		print(sess.run(state))'''

#Fetch 同时运行多个OP 
'''input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(1.0)

add = tf.add(input2,input3)
mul = tf.multiply(input1,add)

with tf.Session() as sess:
	result = sess.run([mul,add]) #可以运行多个OP
	print (result)'''

#Feed 
'''input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
#feed的数据以字典的形式传入
with tf.Session() as sess:
	print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))'''


#讲解一个简单实例
'''import numpy as np 
x_data = np.random.rand(100)
y_data = x_data*0.1+0.2
#构建一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data+b

#二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
#定义最小化代价函数
train = optimizer.minimize(loss)
#初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for step in range(1000):
		sess.run(train)
		if step%20 ==0:
			print(step,sess.run([k,b]))'''
