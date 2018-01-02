import tensorflow as tf
import numpy as np
import globalV
from darknet import *


class MyModel (object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, name='inputImage', shape=[None, globalV.FLAGS.height, globalV.FLAGS.width, 3])
        processX = tf.subtract(tf.multiply(tf.divide(self.x, 255), 2.0), 1.0)
        self.y = tf.placeholder(tf.int64, name='outputClass', shape=[None])
        self.isTraining = tf.placeholder(tf.bool)
        core_net = darknet19(processX, is_training=self.isTraining)

        # Last layer network
        w = tf.get_variable(name='weight', shape=[1000, globalV.FLAGS.numClass], dtype=tf.float32)
        self.b = tf.get_variable(name='bias', shape=[globalV.FLAGS.numClass], dtype=tf.float32)

        # Softmax loss function and Adam optimizer
        predictY = tf.add(tf.matmul(core_net, w), self.b)
        self.meanLoss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(self.y, globalV.FLAGS.numClass), logits=predictY))
        optimizer = tf.train.AdamOptimizer(globalV.FLAGS.lr)
        self.trainStep = optimizer.minimize(self.meanLoss)

        # Predict the output
        correctPrediction = tf.equal(tf.argmax(predictY, 1), self.y)
        self.predictIndex = tf.argmax(predictY, 1)
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        # Log all output
        self.averageL = tf.placeholder(tf.float32)
        lossSummary = tf.summary.scalar("averageLoss", self.averageL)
        self.averageA = tf.placeholder(tf.float32)
        accSummary = tf.summary.scalar("averageAcc", self.averageA)

        # Merge all log
        self.merged = tf.summary.merge_all()

        # Initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if globalV.FLAGS.NEW == 1:
            if tf.gfile.Exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/logs'):
                tf.gfile.DeleteRecursively(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/logs')
            tf.gfile.MakeDirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/logs')
        self.trainWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/logs/train', self.sess.graph)
        self.testWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/logs/test')

        # Start point and log
        self.start = 1
        self.check = 10


    def restoreModel(self):
        self.saver.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model/model.ckpt')
        npzFile = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model/checkpoint.npz')
        self.start = npzFile['start']
        self.check = npzFile['check']


    def trainDarknet(self, trainX, trainY):
        if globalV.FLAGS.NEW == 0:
            self.restoreModel()
        for i in range(self.start, globalV.FLAGS.maxSteps):
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
            # # Validation
            # losses = []
            # accuracies = []
            # for j in range(0, valX.shape[0], globalV.FLAGS.batchSize):
            #     xBatch = valX[j:j + globalV.FLAGS.batchSize]
            #     yBatch = valY[j:j + globalV.FLAGS.batchSize]
            #     valLoss, valAccuracy, _ = sess.run([meanLoss, accuracy, trainStep],feed_dict={x: xBatch, y: yBatch, isTraining: 0})
            #     losses.append(valLoss)
            #     accuracies.append(valAccuracy)
            #
            # feed = {averageL: sum(losses) / len(losses), averageA: sum(accuracies) / len(accuracies)}
            # summary = sess.run(merged, feed_dict=feed)
            # testWriter.add_summary(summary, i)
            if i % self.check == 0:
                savePath = self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model/model.ckpt')
                np.savez(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/model/checkpoint.npz', start=i, check=self.check)
                print('Model saved in file: {0}'.format(savePath))
                if i / self.check == 10:
                    self.check *= 10

        # Close Log
        self.trainWriter.close()
        self.testWriter.close()