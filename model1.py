import tensorflow as tf
import numpy as np
import globalV
from darknet import *


class model1 (object):
    def __init__(self):
        # All placeholder
        self.x = tf.placeholder(tf.float32, name='inputImage', shape=[None, globalV.FLAGS.height, globalV.FLAGS.width, 3])
        self.ySeen = tf.placeholder(tf.int64, name='outputClassIndexSeen', shape=[None])
        self.attYSeen = tf.placeholder(tf.float32, name='outputClassAttSeen', shape=[None, globalV.FLAGS.numAtt])
        self.yUnseen = tf.placeholder(tf.int64, name='outputClassIndexUnseen', shape=[None])
        self.attYUnseen = tf.placeholder(tf.float32, name='outputClassAttUnseen', shape=[None, globalV.FLAGS.numAtt])
        self.allClassAtt = tf.placeholder(tf.float32, name='allClassAtt', shape=[None, globalV.FLAGS.numAtt])
        self.isTraining = tf.placeholder(tf.bool)

        # Transform feature to same size as attribute (Linear compatibility)
        wF_A = tf.get_variable(name='wF_A', shape=[1000, globalV.FLAGS.numAtt], dtype=tf.float32)
        bF_A = tf.get_variable(name='bF_A', shape=[globalV.FLAGS.numAtt], dtype=tf.float32)

        # From attribute to hidden
        wH = tf.get_variable(name='wH', shape=[globalV.FLAGS.numAtt, globalV.FLAGS.numHid], dtype=tf.float32)
        bH = tf.get_variable(name='bH', shape=[globalV.FLAGS.numHid], dtype=tf.float32)

        # From hidden to softmax
        wS = tf.get_variable(name='wS', shape=[globalV.FLAGS.numHid, globalV.FLAGS.numClass], dtype=tf.float32)
        bS = tf.get_variable(name='bS', shape=[globalV.FLAGS.numClass], dtype=tf.float32)

        # Input -> feature extract from CNN darknet
        processX = tf.subtract(tf.multiply(tf.divide(self.x, 255), 2.0), 1.0)
        self.coreNet = darknet19(processX, is_training=self.isTraining)

        # Feature -> Attribute
        self.outAtt = tf.add(tf.matmul(self.coreNet, wF_A), bF_A)

        # Attribute -> Hidden
        outHidSeen = tf.sigmoid(tf.add(tf.matmul(self.outAtt, wH), bH))
        outHidUnseen = tf.sigmoid(tf.add(tf.matmul(self.attYUnseen, wH), bH))

        # Hidden -> Softmax
        outSoftSeen = tf.add(tf.matmul(outHidSeen, wS), bS)
        outSoftUnseen = tf.add(tf.matmul(outHidUnseen, wS), bS)

        # Softmax Loss for seen class
        self.lossSoftSeen = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(self.ySeen, globalV.FLAGS.numClass), logits=outSoftSeen))

        # Softmax Los for unseen class
        self.lossSoftUnSeen = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(self.yUnseen, globalV.FLAGS.numClass), logits=outSoftUnseen))

        # Dot-Product similarity loss for seen class
        # outAtt == self.attYSeen
        # self.lossAttSeen = tf.reduce_mean(tf.squared_difference(outAtt, self.attYSeen))
        # Denominator (|A||B|)
        imageValue = tf.expand_dims(tf.reduce_sum(tf.square(self.outAtt), axis=1), axis=1)
        classValue = tf.expand_dims(tf.reduce_sum(tf.square(self.allClassAtt), axis=1), axis=0)
        prodValue = tf.multiply(imageValue, classValue)
        # Numerator (A.B)
        imageFeatures = tf.expand_dims(self.outAtt, axis=1)
        dotProduct = tf.reduce_sum(tf.multiply(imageFeatures, self.allClassAtt), axis=2)
        dotProduct = tf.divide(dotProduct, prodValue)
        # Cost Function
        correctClass = tf.tile(tf.reduce_sum(tf.multiply(dotProduct, tf.one_hot(self.ySeen, globalV.FLAGS.numClass)), axis=1, keep_dims=True), [1, globalV.FLAGS.numClass])
        self.log1 = dotProduct
        self.log2 = correctClass
        cost = 1.0 - correctClass + dotProduct
        self.log3 = cost
        cost = tf.maximum(cost, 0.0)
        cost = tf.multiply(cost, tf.one_hot(self.ySeen, globalV.FLAGS.numClass, on_value=0.0, off_value=1.0))
        self.log4 = cost
        cost = tf.reduce_sum(cost, axis=1)
        self.lossAttSeen = tf.reduce_mean(cost)

        # Total Loss
        # self.totalLoss = self.lossSoftSeen + self.lossSoftUnSeen + self.lossAttSeen
        self.totalLoss = self.lossAttSeen

        # Define Optimizer
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            self.trainStep = tf.train.AdamOptimizer(globalV.FLAGS.lr).minimize(self.totalLoss)

        # Calculate accuracy
        # self.predictIndex = tf.argmax(outSoftSeen, 1)
        self.predictIndex = tf.argmax(dotProduct, 1)
        correctPrediction = tf.equal(self.predictIndex, self.ySeen)
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        # Log each loss
        self.avgLossSoftSeen = tf.placeholder(tf.float32); tf.summary.scalar("avgLossSoftSeen", self.avgLossSoftSeen)
        self.avgLossAttSeen = tf.placeholder(tf.float32); tf.summary.scalar("avgLossAttSeen", self.avgLossAttSeen)
        self.avgLossSoftUnSeen = tf.placeholder(tf.float32); tf.summary.scalar("avgLossSoftUnSeen", self.avgLossSoftUnSeen)

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
            if tf.gfile.Exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model1/logs'):
                tf.gfile.DeleteRecursively(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model1/logs')
            tf.gfile.MakeDirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model1/logs')
        self.trainWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model1/logs/train', self.sess.graph)
        self.testWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model1/logs/test')

        # Start point and log
        self.Start = 1
        self.Check = 10
        if globalV.FLAGS.NEW == 0:
            self.restoreModel()
        else:
            variableToRestore = tf.contrib.slim.get_variables_to_restore(exclude=['wF_A', 'bF_A', 'wH', 'bH', 'wS', 'bS'])
            saverDarknet = tf.train.Saver(variableToRestore)
            saverDarknet.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/model/model.ckpt')


    def restoreModel(self):
        self.saver.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model1/model/model.ckpt')
        npzFile = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model1/model/checkpoint.npz')
        self.Start = npzFile['Start']
        self.Check = npzFile['Check']

    import numpy
    numpy.set_printoptions(threshold=numpy.inf)
    def trainAtt(self, seenX, seenY, seenAttY, unseenX, unseenY, unseenAttY, allClassAtt):
        for i in range(self.Start, globalV.FLAGS.maxSteps + 1):
            print('Loop {0}/{1}'.format(i, globalV.FLAGS.maxSteps))
            maxAmount = np.minimum(seenX.shape[0], unseenX.shape[0])
            s = np.arange(maxAmount)
            np.random.shuffle(s)

            # Shuffle data
            seenX = seenX[s]; seenY = seenY[s]; seenAttY = seenAttY[s]
            unseenX = unseenX[s]; unseenY = unseenY[s]; unseenAttY = unseenAttY[s]

            # Train
            losses = []
            accuracies = []
            l1s = []; l2s = []; l3s = []
            for j in range(0, maxAmount, globalV.FLAGS.batchSize):
                xBatchSeen = seenX[j:j + globalV.FLAGS.batchSize]
                yBatchSeen = seenY[j:j + globalV.FLAGS.batchSize]
                attYBatchSeen = seenAttY[j:j + globalV.FLAGS.batchSize]
                yBatchUnseen = unseenY[j:j + globalV.FLAGS.batchSize]
                attYBatchUnseen = unseenAttY[j:j + globalV.FLAGS.batchSize]


                l1, l2, l3, trainLoss, trainAccuracy,  _ = self.sess.run([self.lossSoftSeen, self.lossAttSeen, self.lossSoftUnSeen, self.totalLoss, self.accuracy ,self.trainStep], feed_dict={self.x: xBatchSeen, self.ySeen: yBatchSeen, self.attYSeen: attYBatchSeen,
                                                                                                                         self.yUnseen: yBatchUnseen, self.attYUnseen: attYBatchUnseen, self.allClassAtt: allClassAtt,
                                                                                                                         self.isTraining: 1})
                losses.append(trainLoss)
                accuracies.append(trainAccuracy)
                l1s.append(l1); l2s.append(l2); l3s.append(l3)

            feed = {self.averageL: sum(losses) / len(losses), self.averageA: sum(accuracies) / len(accuracies),
                    self.avgLossSoftSeen: sum(l1s) / len(l1s), self.avgLossAttSeen: sum(l2s) / len(l2s), self.avgLossSoftUnSeen: sum(l3s) / len(l3s)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.trainWriter.add_summary(summary, i)

            # Validation
            losses = []
            accuracies = []
            l1s = []
            for j in range(0, unseenX.shape[0], globalV.FLAGS.batchSize):
                xBatch = unseenX[j:j + globalV.FLAGS.batchSize]
                yBatch = unseenY[j:j + globalV.FLAGS.batchSize]
                l1, valLoss, valAccuracy = self.sess.run([self.lossAttSeen, self.lossSoftSeen, self.accuracy], feed_dict={self.x: xBatch, self.ySeen: yBatch, self.allClassAtt: allClassAtt, self.isTraining: 0})
                losses.append(valLoss)
                accuracies.append(valAccuracy)
                l1s.append(l1)
            feed = {self.averageL: 0, self.averageA: sum(accuracies) / len(accuracies),
                    self.avgLossSoftSeen: sum(losses) / len(losses), self.avgLossAttSeen: sum(l1s) / len(l1s), self.avgLossSoftUnSeen: 0}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.testWriter.add_summary(summary, i)

            if i % self.Check == 0:
                savePath = self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model1/model/model.ckpt',global_step=i)
                print('Model saved in file: {0}'.format(savePath))
                if i / self.Check == 10:
                    self.Check *= 10

            # Save State
            np.savez(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model1/model/checkpoint.npz', Start=i+1, Check=self.Check)
            self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model1/model/model.ckpt')


    def predict(self, trainX, allClassAtt):
        return self.sess.run(self.predictIndex, feed_dict={self.x: trainX, self.allClassAtt: allClassAtt, self.isTraining: 0})

    def getFeature(self, trainX):
        return self.sess.run(self.coreNet, feed_dict={self.x: trainX, self.isTraining: 0})

    def getAttribute(self, trainX):
        return self.sess.run(self.outAtt, feed_dict={self.x: trainX, self.isTraining: 0})

