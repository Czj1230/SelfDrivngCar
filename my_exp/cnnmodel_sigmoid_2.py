import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy

#产生随机变量
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#产生二维卷积层
def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

#产生池化层
def max_pool(x,stride):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,stride,stride,1], padding='SAME') #kernel size 2x2, stride

# define placeholder for inputs to network
x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = x
# print(x_image.shape)  # [n_samples, 66,200,3]

#first convolutional layer
W_conv1 = weight_variable([5, 5, 3, 12]) # patch 5x5, in size 3, out size 24, 也即5x5x3的卷积核有24个
b_conv1 = bias_variable([12])

h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1, 2) + b_conv1)

#second convolutional layer
W_conv2 = weight_variable([5, 5, 12, 24])
b_conv2 = bias_variable([24])

h_conv2 = tf.nn.sigmoid(conv2d(h_conv1, W_conv2, 2) + b_conv2)

#third convolutional layer
W_conv3 = weight_variable([5, 5, 24, 48])
b_conv3 = bias_variable([48])

h_conv3 = tf.nn.sigmoid(conv2d(h_conv2, W_conv3, 2) + b_conv3)

#fourth convolutional layer
W_conv4 = weight_variable([3, 3, 48, 64])
b_conv4 = bias_variable([64])

h_conv4 = tf.nn.sigmoid(conv2d(h_conv3, W_conv4, 1) + b_conv4)

#fifth convolutional layer
W_conv5 = weight_variable([3, 3, 64, 64])
b_conv5 = bias_variable([64])

h_conv5 = tf.nn.sigmoid(conv2d(h_conv4, W_conv5, 1) + b_conv5)

#FCL 1
W_fc1 = weight_variable([1152, 1024])
b_fc1 = bias_variable([1024])

h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#FCL 2
W_fc2 = weight_variable([1024, 256])
b_fc2 = bias_variable([256])

h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#FCL 3
W_fc3 = weight_variable([256, 64])
b_fc3 = bias_variable([64])

h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

#FCL 4
W_fc4 = weight_variable([64, 8])
b_fc4 = bias_variable([8])

h_fc4 = tf.nn.sigmoid(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

#Output
W_fc5 = weight_variable([8, 1])
b_fc5 = bias_variable([1])

y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output
