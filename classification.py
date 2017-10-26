import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

mnist=input_data.r
def add_layer(inputs, in_size, out_size, activation_functon=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_functon is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_functon(Wx_plus_b)
    return outputs