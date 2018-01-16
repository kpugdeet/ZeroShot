import tensorflow as tf
import numpy as np
import globalV
from darknet import *


class myModel (object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, name='inputImage', shape=[None, globalV.FLAGS.height, globalV.FLAGS.width, 3])
        processX = tf.subtract(tf.multiply(tf.divide(self.x, 255), 2.0), 1.0)
        self.y = tf.placeholder(tf.int64, name='outputClass', shape=[None])
        self.attY = tf.placeholder(tf.float32, name='outputAtt', shape=[None, globalV.FLAGS.numAtt])
        self.transformW = tf.placeholder(tf.float32, name='transformW', shape=[globalV.FLAGS.numAtt, None])
        self.isTraining = tf.placeholder(tf.bool)
        self.core_net = darknet19(processX, is_training=self.isTraining)

        # Attribute Layer
        wAtt = tf.get_variable(name='attWeight', shape=[1000, globalV.FLAGS.numAtt], dtype=tf.float32)
        bAtt = tf.get_variable(name='attBias', shape=[globalV.FLAGS.numAtt], dtype=tf.float32)
        self.predAtt = tf.sigmoid(tf.add(tf.matmul(self.core_net, wAtt), bAtt))
        attLoss = tf.losses.absolute_difference(self.predAtt, self.attY)
        meanAttLoss = tf.reduce_mean(attLoss)

        # Class Loss at the last layer
        predClass = tf.matmul(self.predAtt, self.transformW)
        meanLastLoss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(self.y, globalV.FLAGS.numClass), logits=predClass))

        # Predict the output for darknet
        self.predictIndex = tf.argmax(predClass, 1)


        correctPrediction = tf.equal(tf.argmax(predClass, 1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        self.finalLoss = meanAttLoss

        # Batch Normalization fixed
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            self.attTrainStep = tf.train.AdamOptimizer(globalV.FLAGS.lr).minimize(self.finalLoss)

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
            if tf.gfile.Exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/myModel/logs'):
                tf.gfile.DeleteRecursively(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/myModel/logs')
            tf.gfile.MakeDirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/myModel/logs')
        self.trainWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/myModel/logs/train', self.sess.graph)
        self.testWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/myModel/logs/test')

        # Start point and log
        self.Start = 1
        self.Check = 10
        if globalV.FLAGS.NEW == 0:
            self.restoreModel()
        else:
            variableToRestore = tf.contrib.slim.get_variables_to_restore(exclude=['attBias', 'attWeight'])
            saverDarknet = tf.train.Saver(variableToRestore)
            saverDarknet.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/model/model.ckpt')


    def restoreModel(self):
        self.saver.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/myModel/model/model.ckpt')
        npzFile = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/myModel/model/checkpoint.npz')
        self.Start = npzFile['Start']
        self.Check = npzFile['Check']


    def trainAtt(self, trainX, trainY, trainYAtt, valX, valY, valYAtt, transformW):
        print(trainX.shape, trainY.shape, trainYAtt.shape, valX.shape, valY.shape, valYAtt.shape, transformW.shape)
        for i in range(self.Start, globalV.FLAGS.maxSteps + 1):
            print('Loop {0}/{1}'.format(i, globalV.FLAGS.maxSteps))
            # Train
            s = np.arange(trainX.shape[0])
            np.random.shuffle(s)
            tmpX = trainX[s]
            tmpY = trainY[s]
            tmpAttY = trainYAtt[s]
            losses = []
            accuracies = []
            for j in range(0, trainX.shape[0], globalV.FLAGS.batchSize):
                xBatch = tmpX[j:j + globalV.FLAGS.batchSize]
                yBatch = tmpY[j:j + globalV.FLAGS.batchSize]
                attYBatch = tmpAttY[j:j + globalV.FLAGS.batchSize]
                trainLoss, trainAccuracy,  _ = self.sess.run([self.finalLoss, self.accuracy ,self.attTrainStep], feed_dict={self.x: xBatch, self.y: yBatch, self.attY: attYBatch,
                                                                                                                            self.transformW: transformW, self.isTraining: 1})
                losses.append(trainLoss)
                accuracies.append(trainAccuracy)
            feed = {self.averageL: sum(losses) / len(losses), self.averageA: sum(accuracies) / len(accuracies)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.trainWriter.add_summary(summary, i)

            # Validation
            losses = []
            accuracies = []
            for j in range(0, valX.shape[0], globalV.FLAGS.batchSize):
                xBatch = valX[j:j + globalV.FLAGS.batchSize]
                yBatch = valY[j:j + globalV.FLAGS.batchSize]
                attYBatch = valYAtt[j:j + globalV.FLAGS.batchSize]
                valLoss, valAccuracy, _ = self.sess.run([self.finalLoss, self.accuracy, self.attTrainStep], feed_dict={self.x: xBatch, self.y: yBatch, self.attY: attYBatch,
                                                                                                                       self.transformW: transformW, self.isTraining: 0})
                losses.append(valLoss)
                accuracies.append(valAccuracy)
            feed = {self.averageL: sum(losses) / len(losses), self.averageA: sum(accuracies) / len(accuracies)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.testWriter.add_summary(summary, i)

            if i % self.Check == 0:
                savePath = self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/myModel/model/model.ckpt',global_step=i)
                print('Model saved in file: {0}'.format(savePath))
                if i / self.Check == 10:
                    self.Check *= 10

            # Save State
            np.savez(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/myModel/model/checkpoint.npz', Start=i+1, Check=self.Check)
            self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/myModel/model/model.ckpt')

    def checkCoreNet(self, trainX):
        return self.sess.run(self.core_net, feed_dict={self.x: trainX, self.isTraining: 0})

    def predict(self, trainX, transformW):
        return self.sess.run(self.predictIndex, feed_dict={self.x: trainX, self.transformW: transformW, self.isTraining: 0})

    def calAcc(self, trainX, trainY, transformW):
        accuracies = []
        for j in range(0, trainX.shape[0], globalV.FLAGS.batchSize):
            xbatch = trainX[j:j + globalV.FLAGS.batchSize]
            yBatch = trainY[j:j + globalV.FLAGS.batchSize]
            acc = self.sess.run(self.accuracy, feed_dict={self.x: xbatch, self.y: yBatch, self.transformW: transformW, self.isTraining: 0})
            accuracies.append(acc)
        return sum(accuracies) / len(accuracies)

    def predAtts(self, trainX):
        atts = None
        for j in range(0, trainX.shape[0], globalV.FLAGS.batchSize):
            xBatch = trainX[j:j + globalV.FLAGS.batchSize]
            att = self.sess.run(self.predAtt, feed_dict={self.x: xBatch, self.isTraining: 0})
            if atts is None:
                atts = att
            else:
                atts = np.concatenate((atts, att), axis=0)
        return np.array(atts)
