from rbm import RBM
import tensorflow as tf
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './mnistData/', 'Directory for storing data')

# First RBM
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
rbmobject1 = RBM(784, 100, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.001)


# Train First RBM
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(10)
  cost = rbmobject1.partial_fit(batch_xs)
  print(cost)

print("###########################################")
batch_xs, batch_ys = mnist.test.next_batch(10)
cost = rbmobject1.partial_fit(batch_xs)
print(cost)