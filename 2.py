import tensorflow as tf 
import numpy as np
sess = tf.Session()
'''x_vals = np.array([1.,2.,3.,4.,5.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)
my_product =tf.multiply(x_data,m_const)
for x_val in x_vals:
	print(sess.run(my_product,feed_dict={x_data:x_val}))''' #运行时将占位符填上
# 首先创建数据和占位符：
my_array = np.array([[1.,3.,5.,7.,9.],
					[-2.,0.,2.,4.,6.],
					[-6.,-3.,0.,3.,6.]])
x_vals = np.array([my_array,my_array+1])
x_data = tf.placeholder(tf.float32,shape=(3,5))
# 创建矩阵乘法和加法中要用到的常量矩阵：
m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])
# 声明操作，表示成计算图：
prod1 = tf.matmul(x_data,m1)
prod2 = tf.matmul(prod1,m2)
add1 = tf.add(prod2,a1)
#最后通过计算图赋值
for x_val in x_vals:
	print(sess.run(add1,feed_dict={x_data:x_val}))

#接下来，如何传播数据的多层layer
