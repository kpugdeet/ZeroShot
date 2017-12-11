import tensorflow as tf
import numpy as np
import globalV

# Add summary to tensorboard
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

class vec2att(object):

    def __init__(self, xDim, yDim):
        self.x = tf.placeholder(tf.float64, name='nameVec', shape=[None, xDim])
        self.y = tf.placeholder(tf.float64, name='att', shape=[None, yDim])
        with tf.name_scope('w'):
            w = tf.get_variable(name='w', shape=[xDim, 100], dtype=tf.float64)
            variable_summaries(w)
        with tf.name_scope('w1'):
            w1 = tf.get_variable(name='w1', shape=[100, yDim], dtype=tf.float64)
            variable_summaries(w1)
        with tf.name_scope('b'):
            b = tf.get_variable(name='b', shape=[100], dtype=tf.float64)
            variable_summaries(b)
        with tf.name_scope('b1'):
            b1 = tf.get_variable(name='b1', shape=[yDim], dtype=tf.float64)
            variable_summaries(b1)

        hidden =  tf.sigmoid(tf.add(tf.matmul(self.x, w), b))
        self.predictY = tf.sigmoid(tf.add(tf.matmul(hidden, w1), b1))
        tf.summary.histogram('predictY', self.predictY)

        # Define loss
        totalLoss = tf.squared_difference(self.predictY, self.y)
        self.meanLoss = tf.reduce_mean(totalLoss) + globalV.FLAGS.reg * tf.nn.l2_loss(w)
        tf.summary.scalar('meanLoss', self.meanLoss)
        optimizer = tf.train.AdamOptimizer(globalV.FLAGS.lr)
        self.trainStep = optimizer.minimize(self.meanLoss)
        self.merged = tf.summary.merge_all()

        # Initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if tf.gfile.Exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/logs'):
            tf.gfile.DeleteRecursively(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/logs')
        tf.gfile.MakeDirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/logs')
        self.trainWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/logs/train', self.sess.graph)
        self.testWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/logs/test')

    def train(self, trainVec, trainAtt, testVec, testAtt, verbose=False):
        for i in range(globalV.FLAGS.maxSteps):
            if i % 10 == 0:
                summary, testLoss, _ = self.sess.run([self.merged, self.meanLoss, self.trainStep], feed_dict={self.x: testVec, self.y: testAtt})
                self.testWriter.add_summary(summary, i)
            summary, trainLoss, _ = self.sess.run([self.merged, self.meanLoss, self.trainStep], feed_dict={self.x: trainVec, self.y: trainAtt})
            self.trainWriter.add_summary(summary, i)
            if i % 10 == 0 and verbose:
                print('Loss at step: {0:<5} tr:{1:<15.5F} te:{2:.5F}'.format(i, trainLoss, testLoss))
        self.trainWriter.close()
        self.testWriter.close()

    def predict(self, nameVec):
        output = np.around(self.sess.run(self.predictY, feed_dict={self.x: nameVec}))
        return output