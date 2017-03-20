import argparse
import tensorflow as tf
import sys

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main(_):
    with tf.device('/gpu:0'):
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

        X = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(X,W) + b

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        for i in range(2000):
            batch = mnist.train.next_batch(100)
            train_step.run(feed_dict={X: batch[0], y_:batch[1]})

        correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        print(accuracy.eval(feed_dict={X: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
