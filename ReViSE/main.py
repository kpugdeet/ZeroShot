import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import globalV
import numpy as np
import os
import utils
import tensorflow as tf
from darknet import *
from loadData import loadData
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset Path
    parser.add_argument('--BASEDIR', type=str, default='/media/dataHD3/kpugdeet/', help='Base folder for dataset and logs')
    parser.add_argument('--AWA2PATH', type=str, default='AWA2/Animals_with_Attributes2/', help='Path for AWA2 dataset')
    parser.add_argument('--CUBPATH', type=str, default='CUB/CUB_200_2011/', help='Path for CUB dataset')
    parser.add_argument('--SUNPATH', type=str, default='SUN/SUNAttributeDB/', help='Path for SUN dataset')
    parser.add_argument('--APYPATH', type=str, default='APY/attribute_data/', help='Path for APY dataset')
    parser.add_argument('--GOOGLE', type=str, default='GoogleNews-vectors-negative300.bin', help='Path for google Word2Vec model')
    parser.add_argument('--KEY', type=str, default='APY',help='Choose dataset (AWA2, CUB, SUN, APY)')
    parser.add_argument('--DIR', type=str, default='APY_ReViSE_0', help='Choose working directory')

    # Image size
    parser.add_argument('--width', type=int, default=300, help='Width')
    parser.add_argument('--height', type=int, default=300, help='Height')

    # Hyper Parameter
    parser.add_argument('--maxSteps', type=int, default=1, help='Number of steps to run trainer.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--numClass', type=int, default=32, help='Number of class')
    parser.add_argument('--batchSize', type=int, default=32, help='Batch size')
    parser.add_argument('--numAtt', type=int, default=300, help='Dimension of Attribute')

    # Choose what to do
    parser.add_argument('--OPT', type=int, default=4, help='1.Darknet, 2.Attribute, 3.Classify, 4.Accuracy')
    globalV.FLAGS, _ = parser.parse_known_args()

    # Check Folder exist
    if not os.path.exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR):
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR)
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/revise')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/revise/logs')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/revise/model')

    # Load data
    print('\nLoad Data for {0}'.format(globalV.FLAGS.KEY))
    (trainClass, trainAtt, trainVec, trainX, trainY, trainYAtt), (valClass, valAtt, valVec, valX, valY, valYAtt), (testClass, testAtt, testVec, testX, testY, testYAtt) = loadData.getData()
    if globalV.FLAGS.KEY == 'SUN' or globalV.FLAGS.KEY == 'APY':
        print('       {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format('numClass', 'classAtt', 'classVec','inputX', 'outputY', 'outputAtt'))
        print('Train: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format(len(trainClass), str(trainAtt.shape), str(trainVec.shape), str(trainX.shape), str(trainY.shape), str(trainYAtt.shape)))
        print('Valid: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format(len(valClass), str(valAtt.shape), str(valVec.shape), str(valX.shape), str(valY.shape), str(valYAtt.shape)))
        print('Test:  {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format(len(testClass), str(testAtt.shape), str(testVec.shape), str(testX.shape), str(testY.shape), str(testYAtt.shape)))
    else:
        print('       {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format('numClass', 'classAtt', 'classVec','inputX', 'outputY'))
        print('Train: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format(len(trainClass), str(trainAtt.shape), str(trainVec.shape), str(trainX.shape), str(trainY.shape)))
        print('Valid: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format(len(valClass), str(valAtt.shape), str(valVec.shape), str(valX.shape), str(valY.shape)))
        print('Test:  {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format(len(testClass), str(testAtt.shape), str(testVec.shape), str(testX.shape), str(testY.shape)))

    # Show class name that index with total classes
    def printClassName(pos):
        if pos < len(trainClass):
            return trainClass[pos]
        elif pos < len(trainClass)+len(valClass):
            return valClass[pos-len(trainClass)]
        else:
            return testClass[pos-len(trainClass)-len(valClass)]

    # Attribute Modification
    concatAtt = np.concatenate((trainAtt, valAtt, testAtt), axis=0)
    word2Vec_concatAtt = np.concatenate((trainVec, valVec, testVec), axis=0)
    concatAtt_D = np.concatenate((concatAtt, word2Vec_concatAtt), axis=1)
    concatAtt_D = word2Vec_concatAtt

    # Check where there is some class that has same attributes
    print('\nCheck matching classes attributes')
    for i in range(concatAtt_D.shape[0]):
        for j in range(i + 1, concatAtt_D.shape[0]):
            if np.array_equal(concatAtt_D[i], concatAtt_D[j]):
                print('{0} {1}: {2} {3}'.format(i, printClassName(i), j, printClassName(j)))
    print('')


    # Split train data 70/30 for each class
    trX70 = None; trY70 = None; trAtt70 = None
    trX30 = None; trY30 = None; trAtt30 = None
    for z in range(0, len(trainClass)):
        eachInputX = []
        eachInputY = []
        eachInputAtt = []
        for k in range(0, trainX.shape[0]):
            if trainY[k] == z:
                eachInputX.append(trainX[k])
                eachInputY.append(trainY[k])
                eachInputAtt.append(concatAtt_D[trainY[k]])
        eachInputX = np.array(eachInputX)
        eachInputY = np.array(eachInputY)
        eachInputAtt = np.array(eachInputAtt)
        divEach = int(eachInputX.shape[0] * 0.7)
        if trX70 is None:
            trX70 = eachInputX[:divEach]
            trY70 = eachInputY[:divEach]
            trAtt70 = eachInputAtt[:divEach]
            trX30 = eachInputX[divEach:]
            trY30 = eachInputY[divEach:]
            trAtt30 = eachInputAtt[divEach:]
        else:
            trX70 = np.concatenate((trX70,  eachInputX[:divEach]), axis=0)
            trY70 = np.concatenate((trY70, eachInputY[:divEach]), axis=0)
            trAtt70 = np.concatenate((trAtt70, eachInputAtt[:divEach]), axis=0)
            trX30 = np.concatenate((trX30, eachInputX[divEach:]), axis=0)
            trY30 = np.concatenate((trY30, eachInputY[divEach:]), axis=0)
            trAtt30 = np.concatenate((trAtt30, eachInputAtt[divEach:]), axis=0)

    # Val class
    s = np.arange(valX.shape[0])
    # np.random.shuffle(s)
    tmp = list()
    for i in range(valY.shape[0]):
        tmp.append(concatAtt_D[valY[i]+len(trainClass)])
    vX = valX[s]
    vY = valY[s] + len(trainClass)
    vAtt = np.array(tmp)[s]

    # Test class
    s = np.arange(testX.shape[0])
    # np.random.shuffle(s)
    tmp = list()
    for i in range(testY.shape[0]):
        tmp.append(concatAtt_D[testY[i]+len(trainClass)+len(valClass)])
    teX = testX[s]
    teY = testY[s] + len(trainClass) + len(valClass)
    teAtt = np.array(tmp)[s]

    print('Shuffle Data shape')
    print(trX70.shape, trY70.shape, trAtt70.shape)
    print(trX30.shape, trY30.shape, trAtt30.shape)
    print(vX.shape, vY.shape, vAtt.shape)
    print(teX.shape, teY.shape, teAtt.shape)

    # Get all Classes name and attribute name
    allClassName = np.concatenate((np.concatenate((trainClass, valClass), axis=0), testClass), axis=0)
    with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
        allClassAttName = [line.strip() for line in f]


    # Model Network
    keepProb = tf.placeholder(tf.float32)
    isTraining = tf.placeholder(tf.bool)

    # Supervised learning Image
    xImages = tf.placeholder(tf.float32, name='inputImage', shape=[None, globalV.FLAGS.height, globalV.FLAGS.width, 3])
    processX = tf.subtract(tf.multiply(tf.divide(xImages, 255), 2.0), 1.0)
    coreNet = darknet19(processX, is_training=isTraining)

    w0_1 = tf.Variable(tf.random_normal([1000, 1000]))
    b0_1 = tf.Variable(tf.random_normal([1000]))
    o0_1 = tf.nn.tanh(tf.add(tf.matmul(coreNet, w0_1), b0_1))

    w1_1 = tf.Variable(tf.random_normal([1000, 500]))
    b1_1 = tf.Variable(tf.random_normal([500]))
    o1_1 = tf.nn.tanh(tf.add(tf.matmul(o0_1, w1_1), b1_1))

    w2_1 = tf.Variable(tf.random_normal([500, 100]))
    b2_1 = tf.Variable(tf.random_normal([100]))
    o2_1 = tf.nn.tanh(tf.add(tf.matmul(o1_1, w2_1), b2_1))

    # Transform function
    w3_1 = tf.Variable(tf.random_normal([100, 50]))
    b3_1 = tf.Variable(tf.random_normal([50]))
    output = tf.add(tf.matmul(o2_1, w3_1), b3_1)
    output = tf.nn.dropout(output, keepProb)
    output = tf.nn.l2_normalize(output, 1, epsilon=1e-12)

    # Supervised learning Text
    xTexts = tf.placeholder(tf.float32, [None, globalV.FLAGS.numAtt], name='XTexts_placeholder')
    w1_2 = tf.Variable(tf.random_normal([globalV.FLAGS.numAtt, 100]))
    b1_2 = tf.Variable(tf.random_normal([100]))
    o1_2 = tf.nn.tanh(tf.add(tf.matmul(xTexts, w1_2), b1_2))

    # Transform function
    w2_2 = tf.Variable(tf.random_normal([100, 50]))
    b2_2 = tf.Variable(tf.random_normal([50]))
    output1 = tf.add(tf.matmul(o1_2, w2_2), b2_2)
    output1 = tf.nn.dropout(output1, keepProb)
    output1 = tf.nn.l2_normalize(output1, 1, epsilon=1e-12)

    # Supervised Loss function
    yLabels = tf.placeholder(tf.int32, [None], name='YLabels_placeholder')
    oneHotClass = tf.placeholder(tf.int32)
    dotProduct = tf.matmul(output, tf.transpose(output1))
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dotProduct, labels=tf.one_hot(yLabels, oneHotClass)))

    # Auto encoder Image
    deW1_1 = tf.Variable(tf.random_normal([100, 500]))
    deB1_1 = tf.Variable(tf.random_normal([500]))
    deO1_1 = tf.nn.tanh(tf.add(tf.matmul(o2_1, deW1_1), deB1_1))
    deW2_1 = tf.Variable(tf.random_normal([500, 1000]))
    deB2_1 = tf.Variable(tf.random_normal([1000]))
    deO2_1 = tf.nn.tanh(tf.add(tf.matmul(deO1_1, deW2_1), deB2_1))

    # Auto encoder Text
    deW1_2 = tf.Variable(tf.random_normal([100, globalV.FLAGS.numAtt]))
    deB1_2 = tf.Variable(tf.random_normal([globalV.FLAGS.numAtt]))
    deO1_2 = tf.nn.tanh(tf.add(tf.matmul(o1_2, deW1_2), deB1_2))

    # Construction Loss
    construction = tf.reduce_mean(tf.square(coreNet - deO2_1)) + tf.reduce_mean(tf.square(xTexts - deO1_2))

    # Cross Modality Distributions Matching
    kernel = utils.gaussian_kernel_matrix
    modality = tf.reduce_mean(kernel(o2_1, o2_1, tf.ones([100], tf.float32)))
    modality += tf.reduce_mean(kernel(o1_2, o1_2, tf.ones([100], tf.float32)))
    modality -= 2 * tf.reduce_mean(kernel(o2_1, o1_2, tf.ones([100], tf.float32)))
    modality = tf.where(modality > 0, modality, 0)

    # Optimizer
    combine = entropy + 1.0 * (construction + 0.1 * modality)
    optimizer = tf.train.AdamOptimizer(1e-2).minimize(combine)

    # Predict output
    predict = tf.argmax(dotProduct, 1)

    # Run Model
    np.set_printoptions(threshold=np.nan, suppress=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    miniBatchSize = globalV.FLAGS.batchSize
    for loop in range(globalV.FLAGS.maxSteps):
        losses = []
        for i in range(0, trX70.shape[0], miniBatchSize):
            xBatch = trX70[i:i + miniBatchSize]
            yBatch = trY70[i:i + miniBatchSize]
            tmp, _, loss = sess.run([dotProduct, optimizer, entropy], feed_dict={xImages: xBatch, yLabels: yBatch, xTexts: concatAtt_D[:15], oneHotClass: 15, keepProb: 0.7, isTraining: 1})
            losses.append(loss)
        totalLoss = sum(losses) / len(losses)
        print("{0} {1}".format(loop,totalLoss))

    # Show output
    predY = []
    for i in range(0, trX70.shape[0], miniBatchSize):
        xBatch = trX70[i:i + miniBatchSize]
        predictTrain_1 = sess.run(predict, feed_dict={xImages: xBatch, xTexts: concatAtt_D[:15], keepProb: 1.0, isTraining: 0})
        predY.extend(predictTrain_1)
    trainAcc_1 = np.mean(np.equal(trY70, predY))
    print(trainAcc_1)

    predY = []
    for i in range(0, trX30.shape[0], miniBatchSize):
        xBatch = trX30[i:i + miniBatchSize]
        predictTrain_1 = sess.run(predict, feed_dict={xImages: xBatch, xTexts: concatAtt_D[:15], keepProb: 1.0, isTraining: 0})
        predY.extend(predictTrain_1)
    trainAcc_1 = np.mean(np.equal(trY30, predY))
    print(trainAcc_1)

    predY = []
    for i in range(0, teX.shape[0], miniBatchSize):
        xBatch = teX[i:i + miniBatchSize]
        predictTrain_1 = sess.run(predict, feed_dict={xImages: xBatch, xTexts: concatAtt_D[20:], keepProb: 1.0, isTraining: 0})
        predY.extend(predictTrain_1)
    trainAcc_1 = np.mean(np.equal(teY-20, predY))
    print(trainAcc_1)













