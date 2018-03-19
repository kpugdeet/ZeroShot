import argparse
import globalV
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from loadData import loadData
from myModel import myModel
from model1 import model1
from darknetModel import darknetModel
from vec2att import vec2att
import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--BASEDIR', type=str, default='/media/dataHD3/kpugdeet/', help='Base folder for dataset and logs')
    parser.add_argument('--AWA2PATH', type=str, default='AWA2/Animals_with_Attributes2/', help='Path for AWA2 dataset')
    parser.add_argument('--CUBPATH', type=str, default='CUB/CUB_200_2011/', help='Path for CUB dataset')
    parser.add_argument('--SUNPATH', type=str, default='SUN/SUNAttributeDB/', help='Path for SUN dataset')
    parser.add_argument('--APYPATH', type=str, default='APY/attribute_data/', help='Path for APY dataset')
    parser.add_argument('--GOOGLE', type=str, default='GoogleNews-vectors-negative300.bin', help='Path for google Word2Vec model')
    parser.add_argument('--KEY', type=str, default='APY',help='Choose dataset (AWA2, CUB, SUN, APY)')
    parser.add_argument('--maxSteps', type=int, default=100, help='Number of steps to run trainer.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--reg', type=float, default=5e4, help='Initial regularization')
    parser.add_argument('--width', type=int, default=300, help='Width')
    parser.add_argument('--height', type=int, default=300, help='Height')
    parser.add_argument('--numClass', type=int, default=15, help='Number of class')
    parser.add_argument('--numAtt', type=int, default=64, help='Dimension of Attribute')
    parser.add_argument('--batchSize', type=int, default=32, help='Number of batch size')
    parser.add_argument('--NEW', type=int, default=1, help='Train from scratch')
    globalV.FLAGS, _ = parser.parse_known_args()

    print('Load Data')
    (trainClass, trainAtt, trainVec, trainX, trainY, trainYAtt), (valClass, valAtt, valVec, valX, valY, valYAtt), (testClass, testAtt, testVec, testX, testY, testYAtt) = loadData.getData()

    if globalV.FLAGS.KEY == 'SUN' or globalV.FLAGS.KEY == 'APY':
        print(len(trainClass), trainAtt.shape, trainVec.shape, trainX.shape, trainY.shape, trainYAtt.shape)
        print(len(valClass), valAtt.shape, valVec.shape, valX.shape, valY.shape, valYAtt.shape)
        print(len(testClass), testAtt.shape, testVec.shape, testX.shape, testY.shape, testYAtt.shape)
    else:
        print(len(trainClass), trainAtt.shape, trainVec.shape, trainX.shape, trainY.shape)
        print(len(valClass), valAtt.shape, valVec.shape, valX.shape, valY.shape)
        print(len(testClass), testAtt.shape, testVec.shape, testX.shape, testY.shape)

    # def printClassName(pos):
    #     if pos < 15:
    #         return trainClass[pos]
    #     elif pos < 20:
    #         return valClass[pos-15]
    #     else:
    #         return testClass[pos-20]
    #
    # concatAtt = np.concatenate((trainAtt, valAtt, testAtt), axis=0)
    # concatOut = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,30, 31])
    # checkAtt = concatAtt
    # for i in range(checkAtt.shape[0]):
    #     for j in range(i + 1, checkAtt.shape[0]):
    #         if np.array_equal(checkAtt[i], checkAtt[j]):
    #             print('{0} {1}: {2} {3}'.format(i, printClassName(i), j, printClassName(j)))
    #             print(checkAtt[i])

    # Paper Way
    def calAccPaper(yAtt, att, y, name='Train'):
        print(yAtt.shape, att.shape)
        yAtt = np.expand_dims(yAtt, axis=1)
        tmpOut = np.divide(yAtt, att, out=np.ones((yAtt.shape[0], att.shape[0], att.shape[1])), where=att!=0)
        tmpOut = np.prod(tmpOut, axis=2)
        acc = np.mean(np.equal(np.argmax(tmpOut, 1), y))
        print(name+' acc: {0}'.format(acc))

    # Dot Prod Way
    def calAccDotProd(yAtt, att, y, name='Train'):
        tmpOut = np.dot(yAtt, att.T)
        acc = np.mean(np.equal(np.argmax(tmpOut, 1), y))
        print(name + ' acc: {0}'.format(acc))

    # Dot Prod Way with Avg
    def calAccDotProdAvg(yAtt, att, y, name='Train'):
        tmpOut = np.dot(yAtt, att.T)/np.sum(att.T, axis=0)
        acc = np.mean(np.equal(np.argmax(tmpOut, 1), y))
        print(name + ' acc: {0}'.format(acc))

    def callAll (yAtt, att, y, name='Train'):
        calAccPaper(yAtt, att, y, name)
        calAccDotProd(yAtt, att, y, name)
        calAccDotProdAvg(yAtt, att, y, name)

    print('Train CNN')
    s = np.arange(trainX.shape[0])
    dividePoint = int(trainX.shape[0]*0.7)
    np.random.shuffle(s)
    tmpX = trainX[s]
    tmpY = trainY[s]
    # tmpYAtt = trainYAtt[s]
    trainYAtt2 = list()
    for i in range(trainY.shape[0]):
        trainYAtt2.append(trainAtt[trainY[i]])
    tmpYAtt = np.array(trainYAtt2)[s]

    # darknet = darknetModel()
    # darknet.trainDarknet(tmpX[:dividePoint], tmpY[:dividePoint], tmpX[dividePoint:], tmpY[dividePoint:])

    # my = myModel()
    # my.trainAtt(tmpX[:dividePoint], tmpY[:dividePoint], tmpYAtt[:dividePoint], tmpX[dividePoint:], tmpY[dividePoint:], tmpYAtt[dividePoint:], np.ceil(trainAtt.T))

    model = model1()
    model.trainAtt(tmpX[:dividePoint], tmpY[:dividePoint], tmpYAtt[:dividePoint], tmpX[dividePoint:], tmpY[dividePoint:],tmpYAtt[dividePoint:], np.ceil(trainAtt.T))

    # # print(my.predict(testX[:50], np.ceil(testAtt.T)))
    # # print(testY[:50])
    # callAll(my.predAtts(trainX), trainAtt, trainY, 'Train')
    # callAll(my.predAtts(valX), valAtt, valY, 'Val')
    # callAll(my.predAtts(testX), testAtt, testY, 'Test')
    # print()

    # # Attribute itself
    # callAll(trainAtt, trainAtt, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]), 'Train')
    # callAll(valAtt, valAtt, np.array([0, 1, 2, 3, 4]), 'Val')
    # callAll(testAtt, testAtt, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), 'Test')
    # print()

    # tmpPredClass = np.dot(testYAtt[:20], np.ceil(testAtt.T))
    # tmpPredictIndex = np.argmax(tmpPredClass, 1)
    # # predictLabel = my.predict(testX[:10], np.ceil(testAtt.T))
    # print('Save image')
    # plt.figure(figsize=(6, 24))
    # for i in range(20):
    #     plt.subplot(10, 2, i+1)
    #     plt.title('P: '+testClass[tmpPredictIndex[i]]+'\nA: '+testClass[testY[i]])
    #     plt.imshow(testX[i], aspect='auto')
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('Test.png')
    # plt.clf()

    # summary_op = tf.summary.image("plot", tmpX)
    # with tf.Session() as sess:
    #     summary = sess.run(summary_op)
    #     writer = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/imageLogs/')
    #     writer.add_summary(summary)
    #     writer.close()

    # s = np.arange(trainX.shape[0])
    # np.random.shuffle(s)
    # tmpX = trainX[s]
    # tmpY = trainY[s]
    # print(model.predict(tmpX[:10]))
    # print(tmpY[:10])


    # print (trainX[0][200])
    #
    # tmp = tf.placeholder(tf.float64, name='nameVec', shape=[None, 448, 448, 3])
    # processTmp = tf.subtract(tf.multiply(tf.divide(tmp, 255), 2.0), 1.0)
    #
    # gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)  # 1/3 memory of total
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpuOptions))
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(processTmp, feed_dict={tmp: trainX[:100]})[0][200])

    # import time
    # time.sleep(10000)
    # # Vector to attribute model
    # model = vec2att(trainVec.shape[1], trainAtt.shape[1])
    # model.train(trainVec, trainAtt, testVec, testAtt, verbose=False)
    # print(trainAtt.shape[1])
    # print(np.mean(np.sum(np.equal(trainAtt, model.predict(trainVec)), axis=1)))
    # print(np.mean(np.sum(np.equal(testAtt, model.predict(testVec)), axis=1)))









