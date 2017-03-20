import argparse
import tensorflow as tf
import sys

from tensorflow.examples.tutorials.mnist import input_data


def conv2dlayer(X, W):
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="SAME")


def conv2d_relu(X, W, b):
    return tf.nn.relu(conv2dlayer(X, W) + b)


def maxpool22(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.02))


def bias(shape):
    return tf.Variable(tf.constant(0.1, tf.float32))


def dense(X, W, b):
    return tf.matmul(X, W) + b


def dense_relu(X, W, b):
    return tf.nn.relu(tf.matmul(X, W) + b)


def dropout(X, p):
    return tf.nn.dropout(X, keep_prob=p)


FLAGS = None


def main(_):
    with tf.device('/cpu:0'):
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        FLAGS.summaries_dir = "/home/mg/Desktop/Notebooks/Playground/summaries"

        with tf.name_scope("input"):
            X = tf.placeholder(tf.float32, [None, 784])
            y_ = tf.placeholder(tf.float32, [None, 10])
            p = tf.placeholder(tf.float32)
            X_image = tf.reshape(X, [-1, 28, 28, 1])

        # Layer 1
        with tf.name_scope("conv_layer1"):
            W1 = weights([5, 5, 1, 32])
            b1 = bias([32])
            conv1 = conv2d_relu(X_image, W1, b1)
            pool1 = maxpool22(conv1)

        # Layer 2
        with tf.name_scope("conv_layer2"):
            W2 = weights([3, 3, 32, 64])
            b2 = bias([64])
            conv2 = conv2d_relu(pool1, W2, b2)
            pool2 = maxpool22(conv2)

        # Layer 3
        with tf.name_scope("fc_layer3"):
            flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])
            W3 = weights([7 * 7 * 64, 1024])
            b3 = bias([1024])
            fc3 = dense_relu(flattened, W3, b3)
            drop1 = dropout(fc3, p)

        # Layer 4
        with tf.name_scope("fc_layer4"):
            W4 = weights([1024, 10])
            b4 = bias([10])
            y = dense(drop1, W4, b4)

        sess = tf.InteractiveSession()

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
            tf.summary.scalar("loss", loss)

        with tf.name_scope("optimizer"):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
            correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        sess.run(tf.global_variables_initializer())

        try:
            for i in range(2000):
                batch = mnist.train.next_batch(100)
                summary, _ = sess.run([merged, train_step], feed_dict={X: batch[0], y_: batch[1], p: 0.5})
                if not i % 100:
                    acc = accuracy.eval(feed_dict={X: batch[0], y_: batch[1], p: 1.0})
                    print("Iteration " + str(i) + ": Training accuracy: " + str(acc))
                train_writer.add_summary(summary, i)

        except:
            pass
        print("Test accuracy: " + str(accuracy.eval(feed_dict={X: mnist.test.images, y_: mnist.test.labels, p: 1.0})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                    help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
