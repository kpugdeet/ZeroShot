import argparse
import globalV
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from loadData import loadData
from vec2att import vec2att
import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--BASEDIR', type=str, default='/media/dataHD3/kpugdeet/', help='Base folder for dataset and logs')
    parser.add_argument('--AWA2PATH', type=str, default='AWA2/Animals_with_Attributes2/', help='Path for AWA2 dataset')
    parser.add_argument('--CUBPATH', type=str, default='CUB/CUB_200_2011/', help='Path for CUB dataset')
    parser.add_argument('--GOOGLE', type=str, default='GoogleNews-vectors-negative300.bin', help='Path for google Word2Vec model')
    parser.add_argument('--KEY', type=str, default='CUB',help='Choose dataset')
    parser.add_argument('--maxSteps', type=int, default=1000, help='Number of steps to run trainer.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--reg', type=float, default=5e4, help='Initial regularization')
    parser.add_argument('--width', type=int, default=448, help='Width')
    parser.add_argument('--height', type=int, default=448, help='Height')
    globalV.FLAGS, _ = parser.parse_known_args()

    print('Load Data')
    (trainClass, trainAtt, trainVec, trainX, trainY), (valClass, valAtt, valVec, valX, valY), (testClass, testAtt, testVec, testX, testY) = loadData.getData()
    print(trainClass)
    print(len(trainClass), trainAtt.shape, trainVec.shape, trainX.shape, trainY.shape)
    print(valClass)
    print(len(valClass), valAtt.shape, valVec.shape, valX.shape, valY.shape)
    print(testClass)
    print(len(testClass), testAtt.shape, testVec.shape, testX.shape, testY.shape)


    # print (trainX[0][200])
    #
    # tmp = tf.placeholder(tf.float64, name='nameVec', shape=[None, 448, 448, 3])
    # processTmp = tf.subtract(tf.multiply(tf.divide(tmp, 255), 2.0), 1.0)
    #
    # gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)  # 1/3 memory of total
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpuOptions))
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(processTmp, feed_dict={tmp: trainX[:100]})[0][200])

    import time
    time.sleep(10000)
    # # Vector to attribute model
    # model = vec2att(trainVec.shape[1], trainAtt.shape[1])
    # model.train(trainVec, trainAtt, testVec, testAtt, verbose=False)
    # print(trainAtt.shape[1])
    # print(np.mean(np.sum(np.equal(trainAtt, model.predict(trainVec)), axis=1)))
    # print(np.mean(np.sum(np.equal(testAtt, model.predict(testVec)), axis=1)))









