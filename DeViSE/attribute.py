import tensorflow as tf
import numpy as np
import globalV
from darknet import *


class attribute (object):
    def __init__(self):
        # All placeholder
        self.x = tf.placeholder(tf.float32, name='inputImage', shape=[None, globalV.FLAGS.height, globalV.FLAGS.width, 3])
        self.ySeen = tf.placeholder(tf.int64, name='outputClassIndexSeen', shape=[None])
        self.attYSeen = tf.placeholder(tf.float32, name='outputClassAttSeen', shape=[None, globalV.FLAGS.numAtt])
        self.allClassAtt = tf.placeholder(tf.float32, name='allClassAtt', shape=[None, globalV.FLAGS.numAtt])
        self.isTraining = tf.placeholder(tf.bool)

        # Transform feature to same size as attribute (Linear compatibility)
        with tf.variable_scope("attribute"):
            wF_H = tf.get_variable(name='wF_H', shape=[1000, 500], dtype=tf.float32)
            bF_H = tf.get_variable(name='bF_H', shape=[500], dtype=tf.float32)
            self.wH_A = tf.get_variable(name='wH_A', shape=[500, globalV.FLAGS.numAtt], dtype=tf.float32)
            bH_A = tf.get_variable(name='bH_A', shape=[globalV.FLAGS.numAtt], dtype=tf.float32)

        # Input -> feature extract from CNN darknet
        processX = tf.subtract(tf.multiply(tf.divide(self.x, 255), 2.0), 1.0)
        self.coreNet = darknet19(processX, is_training=self.isTraining)

        # Feature -> Attribute
        hiddenF = tf.tanh(tf.add(tf.matmul(self.coreNet, wF_H), bF_H))
        self.outAtt = tf.add(tf.matmul(hiddenF, self.wH_A), bH_A)

        # Dot-Product similarity loss for seen class
        # Denominator (|A||B|)
        imageValue = tf.expand_dims(tf.reduce_sum(tf.square(self.outAtt), axis=1), axis=1)
        classValue = tf.expand_dims(tf.reduce_sum(tf.square(self.allClassAtt), axis=1), axis=0)
        prodValue = tf.multiply(imageValue, classValue)
        # Numerator (A.B)
        imageFeatures = tf.expand_dims(self.outAtt, axis=1)
        self.dotProduct = tf.reduce_sum(tf.multiply(imageFeatures, self.allClassAtt), axis=2)
        # (A.B)/(|A||B|)
        # self.dotProduct = tf.divide(self.dotProduct, prodValue)

        # Cost Function
        correctClass = tf.tile(tf.reduce_sum(tf.multiply(self.dotProduct, tf.one_hot(self.ySeen, globalV.FLAGS.numClass)), axis=1, keep_dims=True), [1, globalV.FLAGS.numClass])
        cost = 1.0 - correctClass + self.dotProduct
        cost = tf.maximum(cost, 0.0)
        cost = tf.multiply(cost, tf.one_hot(self.ySeen, globalV.FLAGS.numClass, on_value=0.0, off_value=1.0))
        cost = tf.reduce_sum(cost, axis=1)
        self.totalLoss = tf.reduce_mean(cost)

        # Predict Index
        self.predictIndex = tf.argmax(self.dotProduct, 1)

        # Define Optimizer
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            self.trainStep = tf.train.AdamOptimizer(globalV.FLAGS.lr).minimize(self.totalLoss)

        # Log all output
        self.averageL = tf.placeholder(tf.float32)
        tf.summary.scalar("averageLoss", self.averageL)

        # Merge all log
        self.merged = tf.summary.merge_all()

        # Initialize session
        tfConfig = tf.ConfigProto(allow_soft_placement=True)
        tfConfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfConfig)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)

        # Log directory
        if globalV.FLAGS.TA == 1:
            if tf.gfile.Exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs'):
                tf.gfile.DeleteRecursively(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs')
            tf.gfile.MakeDirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs')
        self.trainWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs/train', self.sess.graph)
        self.valWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs/validate')
        self.testWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs/test')

        # Start point and log
        self.Start = 1
        self.Check = 10
        if globalV.FLAGS.TA == 0:
            self.restoreModel()
        else:
            variables = tf.contrib.slim.get_variables_to_restore()
            variableToRestore = [v for v in variables if v.name.split('/')[0] != 'attribute']
            saverDarknet = tf.train.Saver(variableToRestore)
            saverDarknet.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/darknet/model/model.ckpt')

    def restoreModel(self):
        self.saver.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/model.ckpt')
        npzFile = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/checkpoint.npz')
        self.Start = npzFile['Start']
        self.Check = npzFile['Check']

    def trainAtt(self, seenX, seenY, unseenX, unseenY, unseenX2, unseenY2, allAtt):
        for i in range(self.Start, globalV.FLAGS.maxSteps + 1):
            print('Loop {0}/{1}'.format(i, globalV.FLAGS.maxSteps))

            # Shuffle Data
            s = np.arange(seenX.shape[0])
            np.random.shuffle(s)
            seenX = seenX[s]
            seenY = seenY[s]

            # Train
            losses = []
            for j in range(0, seenX.shape[0], globalV.FLAGS.batchSize):
                xBatchSeen = seenX[j:j + globalV.FLAGS.batchSize]
                yBatchSeen = seenY[j:j + globalV.FLAGS.batchSize]
                trainLoss, _ = self.sess.run([self.totalLoss ,self.trainStep], feed_dict={self.x: xBatchSeen, self.ySeen: yBatchSeen, self.allClassAtt: allAtt, self.isTraining: 1})
                losses.append(trainLoss)

            feed = {self.averageL: sum(losses) / len(losses)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.trainWriter.add_summary(summary, i)

            # Validation
            losses = []
            for j in range(0, unseenX.shape[0], globalV.FLAGS.batchSize):
                xBatch = unseenX[j:j + globalV.FLAGS.batchSize]
                yBatch = unseenY[j:j + globalV.FLAGS.batchSize]
                valLoss = self.sess.run(self.totalLoss, feed_dict={self.x: xBatch, self.ySeen: yBatch, self.allClassAtt: allAtt, self.isTraining: 0})
                losses.append(valLoss)
            feed = {self.averageL: sum(losses) / len(losses)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.valWriter.add_summary(summary, i)

            # Test
            losses = []
            for j in range(0, unseenX2.shape[0], globalV.FLAGS.batchSize):
                xBatch = unseenX2[j:j + globalV.FLAGS.batchSize]
                yBatch = unseenY2[j:j + globalV.FLAGS.batchSize]
                valLoss = self.sess.run(self.totalLoss, feed_dict={self.x: xBatch, self.ySeen: yBatch, self.allClassAtt: allAtt, self.isTraining: 0})
                losses.append(valLoss)
            feed = {self.averageL: sum(losses) / len(losses)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.testWriter.add_summary(summary, i)

            if i % self.Check == 0:
                savePath = self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/model.ckpt',global_step=i)
                print('Model saved in file: {0}'.format(savePath))
                if i / self.Check == 10:
                    self.Check *= 10

            # Save State
            np.savez(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/checkpoint.npz', Start=i+1, Check=self.Check)
            self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/model.ckpt')

    def getLastWeight(self):
        return self.sess.run(self.wH_A)

    def predictClassIndex(self, trainX, allClassAtt):
        outputAtt = None
        for j in range(0, trainX.shape[0], globalV.FLAGS.batchSize):
            xBatch = trainX[j:j + globalV.FLAGS.batchSize]
            if outputAtt is None:
                outputAtt = self.sess.run(self.predictIndex, feed_dict={self.x: xBatch, self.allClassAtt: allClassAtt, self.isTraining: 0})
            else:
                outputAtt = np.concatenate((outputAtt, self.sess.run(self.predictIndex, feed_dict={self.x: xBatch, self.allClassAtt: allClassAtt, self.isTraining: 0})), axis=0)
        return outputAtt

    def getAttribute(self, trainX):
        outputAtt = None
        for j in range(0, trainX.shape[0], globalV.FLAGS.batchSize):
            xBatch = trainX[j:j + globalV.FLAGS.batchSize]
            if outputAtt is None:
                outputAtt = self.sess.run(self.outAtt, feed_dict={self.x: xBatch, self.isTraining: 0})
            else:
                outputAtt = np.concatenate((outputAtt, self.sess.run(self.outAtt, feed_dict={self.x: xBatch, self.isTraining: 0})), axis=0)
        return outputAtt

    def predictScore(self, trainX, allClassAtt):
        outputAtt = None
        for j in range(0, trainX.shape[0], globalV.FLAGS.batchSize):
            xBatch = trainX[j:j + globalV.FLAGS.batchSize]
            if outputAtt is None:
                outputAtt = self.sess.run(self.dotProduct, feed_dict={self.x: xBatch, self.allClassAtt: allClassAtt, self.isTraining: 0})
            else:
                outputAtt = np.concatenate((outputAtt, self.sess.run(self.dotProduct, feed_dict={self.x: xBatch, self.allClassAtt: allClassAtt, self.isTraining: 0})), axis=0)
        return outputAtt

