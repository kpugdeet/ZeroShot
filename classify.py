import tensorflow as tf
import numpy as np
import globalV


class classify (object):
    def __init__(self):
        # All placeholder
        self.y = tf.placeholder(tf.int64, name='outputClassIndexSeen_C', shape=[None])
        self.att = tf.placeholder(tf.float32, name='outputClassAttSeen_C', shape=[None, globalV.FLAGS.numAtt])

        # From attribute to hidden
        wH = tf.get_variable(name='wH_C', shape=[globalV.FLAGS.numAtt, globalV.FLAGS.numHid], dtype=tf.float32)
        bH = tf.get_variable(name='bH_C', shape=[globalV.FLAGS.numHid], dtype=tf.float32)

        # From hidden to softmax
        wS = tf.get_variable(name='wS_C', shape=[globalV.FLAGS.numHid, globalV.FLAGS.numClass], dtype=tf.float32)
        bS = tf.get_variable(name='bS_C', shape=[globalV.FLAGS.numClass], dtype=tf.float32)

        # Attribute -> Hidden
        outHid = tf.sigmoid(tf.add(tf.matmul(self.att, wH), bH))

        # Hidden -> Softmax
        outSoft = tf.add(tf.matmul(outHid, wS), bS)

        # Softmax Loss for seen class
        self.lossSoft = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(self.y, globalV.FLAGS.numClass), logits=outSoft))

        # Total Loss
        self.totalLoss = self.lossSoft

        # Define Optimizer
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            self.trainStep = tf.train.AdamOptimizer(globalV.FLAGS.lr).minimize(self.totalLoss)

        # Calculate accuracy
        self.predictIndex = tf.argmax(outSoft, 1)
        correctPrediction = tf.equal(self.predictIndex, self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        # Log all output
        self.averageL = tf.placeholder(tf.float32)
        tf.summary.scalar("averageLoss", self.averageL)
        self.averageA = tf.placeholder(tf.float32)
        tf.summary.scalar("averageAcc", self.averageA)

        # Merge all log
        self.merged = tf.summary.merge_all()

        # Initialize session
        tfConfig = tf.ConfigProto(allow_soft_placement=True)
        tfConfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfConfig)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)

        # Log directory
        if globalV.FLAGS.NEW == 1:
            if tf.gfile.Exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/classify/logs'):
                tf.gfile.DeleteRecursively(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/classify/logs')
            tf.gfile.MakeDirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/classify/logs')
        self.trainWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/classify/logs/train', self.sess.graph)
        self.testWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/classify/logs/test')

        # Start point and log
        self.Start = 1
        self.Check = 10
        if globalV.FLAGS.NEW == 0:
            self.restoreModel()

    def restoreModel(self):
        self.saver.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/classify/model/model.ckpt')
        npzFile = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/classify/model/checkpoint.npz')
        self.Start = npzFile['Start']
        self.Check = npzFile['Check']

    def trainClassify(self, att, y):
        for i in range(self.Start, globalV.FLAGS.maxSteps + 1):
            print('Loop {0}/{1}'.format(i, globalV.FLAGS.maxSteps))


            trainLoss, trainAccuracy,  _ = self.sess.run([self.totalLoss, self.accuracy, self.trainStep], feed_dict={self.y: y, self.att: att})


            feed = {self.averageL: trainLoss, self.averageA: trainAccuracy}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.trainWriter.add_summary(summary, i)

            if i % self.Check == 0:
                savePath = self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/classify/model/model.ckpt',global_step=i)
                print('Model saved in file: {0}'.format(savePath))
                if i / self.Check == 10:
                    self.Check *= 10

            # Save State
            np.savez(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/classify/model/checkpoint.npz', Start=i+1, Check=self.Check)
            self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/classify/model/model.ckpt')

    def predict(self, att):
        return self.sess.run(self.predictIndex, feed_dict={self.att: att})


