#实现损失函数
#绘图库
import matplotlib.pyplot as plt 
import tensorflow as tf 
x_vals = tf.linspace(-1.,1.,500)
target = tf.constant(0.)
#1 L2正则损失函数(欧拉损失函数：预测值与目标值差值的平方差)
12_y_vals = tf.square(target-x_vals)
12_y_out = sess.run(12_y_vals)
#2 L1正则损失函数(绝对值损失函数) 对差值求绝对值
11_y_vals = tf.abs(target-x_vals)
11_y_out = seaa.run(11_y_vals)
#3 Pseudo-Huber损失函数 是Huber损失函数的连续，平滑估计，试图利用L1和L2正则削弱及支出的陡峭，使目标值附近连续。依赖delta
deltal = tf.constant(0.25)
phuber1_y_vals = tf.mul(tf.square(deltal),tf.sqrt()

