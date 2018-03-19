import tensorflow as tf
from sklearn import datasets
# 声明变量并初始化变量
my_var = tf.Variable(tf.zeros([2,3]))
#激励函数 目的：调节权重和误差 在张量上的非线性操作。
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# 自定义函数
def custom(value):
	return (tf.subtract(3*tf.square(value),value)+10)
print(sess.run(custom(11)))
#激励函数 
#1 整流线性单元 RelU
print(sess.run(tf.nn.relu([-5,-3,3.,10.])))
#2 为了抵消Relu的线性增长部分，ReLU6
print(sess.run(tf.nn.relu6([-5,-3,3,10])))
#3 sigmoid 最常用的连续，平滑函数
print(sess.run(tf.nn.sigmoid([-1.,0.,1.])))
#4 双曲正切函数
print(sess.run(tf.nn.tanh([-1.,0.,1.])))
#5 softsign 
print(sess.run(tf.nn.softsign([-3.,0.,-1.])))
#6 softplus 是ReLU的平滑版
print(sess.run(tf.nn.softplus([-1.,0.,-1.])))
#7 ELU激励与softplus函数相似
print(sess.run(tf.nn.elu([-1.,0.,-1.])))

#开始学习读取数据源
# Iris data 这个数据集？
iris = datasets.load_iris()
print(len(iris.data)) #150个数据集
print(len(iris.target))
print(iris.target[0])
print(set(iris.target))

#进阶



