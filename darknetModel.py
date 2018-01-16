import tensorflow as tf
import numpy as np
import globalV
from darknet import *


class darknetModel (object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, name='inputImage', shape=[None, globalV.FLAGS.height, globalV.FLAGS.width, 3])
        processX = tf.subtract(tf.multiply(tf.divide(self.x, 255), 2.0), 1.0)
        self.y = tf.placeholder(tf.int64, name='outputClass', shape=[None])
        self.isTraining = tf.placeholder(tf.bool)
        self.core_net = darknet19(processX, is_training=self.isTraining)

        # Last layer network
        w = tf.get_variable(name='weight', shape=[1000, globalV.FLAGS.numClass], dtype=tf.float32)
        b = tf.get_variable(name='bias', shape=[globalV.FLAGS.numClass], dtype=tf.float32)

        # Softmax loss function and Adam optimizer
        predictY = tf.add(tf.matmul(self.core_net, w), b)
        self.meanLoss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(self.y, globalV.FLAGS.numClass), logits=predictY))

        # Predict the output for darknet
        correctPrediction = tf.equal(tf.argmax(predictY, 1), self.y)
        self.predictIndex = tf.argmax(predictY, 1)
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        # Batch Normalization fixed
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            self.trainStep = tf.train.AdamOptimizer(globalV.FLAGS.lr).minimize(self.meanLoss)

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
            if tf.gfile.Exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/logs'):
                tf.gfile.DeleteRecursively(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/logs')
            tf.gfile.MakeDirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/logs')
        self.trainWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/logs/train', self.sess.graph)
        self.testWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/logs/test')

        # Start point and log
        self.Start = 1
        self.Check = 10
        if globalV.FLAGS.NEW == 0:
            self.restoreModel()


    def restoreModel(self):
        self.saver.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/model/model.ckpt')
        npzFile = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/model/checkpoint.npz')
        self.Start = npzFile['Start']
        self.Check = npzFile['Check']


    def trainDarknet(self, trainX, trainY, valX, valY):
        print(trainX.shape, trainY.shape, valX.shape, valY.shape)
        for i in range(self.Start, globalV.FLAGS.maxSteps+1):
            print('Loop {0}/{1}'.format(i, globalV.FLAGS.maxSteps))
            # Train
            s = np.arange(trainX.shape[0])
            np.random.shuffle(s)
            tmpX = trainX[s]
            tmpY = trainY[s]
            losses = []
            accuracies = []
            for j in range(0, trainX.shape[0], globalV.FLAGS.batchSize):
                xBatch = tmpX[j:j + globalV.FLAGS.batchSize]
                yBatch = tmpY[j:j + globalV.FLAGS.batchSize]
                trainLoss, trainAccuracy, _ = self.sess.run([self.meanLoss, self.accuracy, self.trainStep], feed_dict={self.x: xBatch, self.y: yBatch, self.isTraining: 1})
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
                valLoss, valAccuracy, _ = self.sess.run([self.meanLoss, self.accuracy, self.trainStep],feed_dict={self.x: xBatch, self.y: yBatch, self.isTraining: 0})
                losses.append(valLoss)
                accuracies.append(valAccuracy)
            feed = {self.averageL: sum(losses) / len(losses), self.averageA: sum(accuracies) / len(accuracies)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.testWriter.add_summary(summary, i)

            if i % self.Check == 0:
                savePath = self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/model/model.ckpt', global_step=i)
                print('Model saved in file: {0}'.format(savePath))
                if i / self.Check == 10:
                    self.Check *= 10

            # Save State
            np.savez(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/model/checkpoint.npz', Start=i+1, Check=self.Check)
            self.saver.save(self.sess,globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/darknet/model/model.ckpt')

        # Close Log
        self.trainWriter.close()
        self.testWriter.close()

    def predict(self, trainX):
        return self.sess.run(self.predictIndex, feed_dict={self.x: trainX, self.isTraining: 0})


    def checkCoreNet(self, trainX):
        return self.sess.run(self.core_net, feed_dict={self.x: trainX, self.isTraining: 0})