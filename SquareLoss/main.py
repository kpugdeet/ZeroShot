import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import globalV
import numpy as np
import os
from loadData import loadData
from attribute import attribute
from classify import classify
from darknetModel import darknetModel
import tensorflow as tf
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
    parser.add_argument('--DIR', type=str, default='APY_0', help='Choose working directory')

    # Image size
    parser.add_argument('--width', type=int, default=300, help='Width')
    parser.add_argument('--height', type=int, default=300, help='Height')

    # Hyper Parameter
    parser.add_argument('--maxSteps', type=int, default=1, help='Number of steps to run trainer.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--numClass', type=int, default=32, help='Number of class')
    parser.add_argument('--batchSize', type=int, default=32, help='Batch size')
    parser.add_argument('--numAtt', type=int, default=64, help='Dimension of Attribute')

    # Initialize or Restore Model
    parser.add_argument('--TD', type=int, default=0, help='Train/Restore Darknet')
    parser.add_argument('--TA', type=int, default=0, help='Train/Restore Attribute')
    parser.add_argument('--TC', type=int, default=0, help='Train/Restore Classify')

    # Choose what to do
    parser.add_argument('--OPT', type=int, default=4, help='1.Darknet, 2.Attribute, 3.Classify, 4.Accuracy')
    globalV.FLAGS, _ = parser.parse_known_args()

    # Check Folder exist
    if not os.path.exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR):
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR)
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/darknet')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/darknet/logs')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/darknet/model')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/logs')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/model')

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
    # word2Vec_concatAtt = np.concatenate((trainVec, valVec, testVec), axis=0)
    # concatAtt = np.concatenate((concatAtt, word2Vec_concatAtt), axis=1)
    concatAtt_D = concatAtt

    # Check where there is some class that has same attributes
    print('\nCheck matching classes attributes')
    for i in range(concatAtt_D.shape[0]):
        for j in range(i + 1, concatAtt_D.shape[0]):
            if np.array_equal(concatAtt_D[i], concatAtt_D[j]):
                print('{0} {1}: {2} {3}'.format(i, printClassName(i), j, printClassName(j)))
    print('')

    # Train class
    trDiv = int(trainX.shape[0] * 0.7)
    s = np.arange(trainX.shape[0])
    np.random.shuffle(s)
    tmp = list()
    for i in range(trainY.shape[0]):
        tmp.append(concatAtt_D[trainY[i]])
    trX = trainX[s]
    trY = trainY[s]
    trAtt = np.array(tmp)[s]

    # Val class
    s = np.arange(valX.shape[0])
    np.random.shuffle(s)
    tmp = list()
    for i in range(valY.shape[0]):
        tmp.append(concatAtt_D[valY[i]+len(trainClass)])
    vX = valX[s]
    vY = valY[s] + len(trainClass)
    vAtt = np.array(tmp)[s]

    # Test class
    s = np.arange(testX.shape[0])
    np.random.shuffle(s)
    tmp = list()
    for i in range(testY.shape[0]):
        tmp.append(concatAtt_D[testY[i]+len(trainClass)+len(valClass)])
    teX = testX[s]
    teY = testY[s] + len(trainClass) + len(valClass)
    teAtt = np.array(tmp)[s]

    print('Shuffle Data shape')
    print(trX.shape, trY.shape, trAtt.shape)
    print(vX.shape, vY.shape, vAtt.shape)
    print(teX.shape, teY.shape, teAtt.shape)

    # Get all Classes name and attribute name
    allClassName = np.concatenate((np.concatenate((trainClass, valClass), axis=0), testClass), axis=0)
    with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
        allClassAttName = [line.strip() for line in f]

    # Train Network
    if globalV.FLAGS.OPT == 1:
        print('\nTrain Darknet')
        darknet = darknetModel()
        darknet.trainDarknet(trX[:trDiv], trY[:trDiv], trX[trDiv:], trY[trDiv:])

    elif globalV.FLAGS.OPT == 2:
        print('\nTrain Attribute')
        attModel = attribute()
        attModel.trainAtt(trX, trAtt, vX, vAtt, teX, teAtt)

    elif globalV.FLAGS.OPT == 3:
        print('\nTrain Classify')
        classifier = classify()
        classifier.trainClassify(concatAtt_D, np.arange(concatAtt_D.shape[0]), 0.5)

    elif globalV.FLAGS.PRE == 4:
        g1 = tf.Graph()
        g2 = tf.Graph()
        with g1.as_default():
            model = attribute()
        with g2.as_default():
            classifier = classify()
        attTmp = model.getAttribute(trX)
        predY = classifier.predict(attTmp)
        print('Train Accuracy = {0:.4f}%'.format(np.mean(np.equal(predY, trY))*100))
        attTmp = model.getAttribute(vX)
        predY = classifier.predict(attTmp)
        print('Val Accuracy = {0:.4f}%'.format(np.mean(np.equal(predY, vY)) * 100))
        attTmp = model.getAttribute(teX)
        predY = classifier.predict(attTmp)
        print('Test Accuracy = {0:.4f}%'.format(np.mean(np.equal(predY, teY)) * 100))

        # Accuracy for each class
        print('')
        # Loop each Train class
        for z in range(0, 15):
            eachInputX = []
            eachInputY = []
            for k in range(trX.shape[0]):
                if trY[k] == z:
                    eachInputX.append(trX[k])
                    eachInputY.append(trY[k])
            eachInputX = np.array(eachInputX)
            eachInputY = np.array(eachInputY)
            attTmp = model.getAttribute(eachInputX)
            predY = classifier.predict(attTmp)
            print('Class: {0:<15} Size: {1:<10} Accuracy = {2:.4f}%'.format(printClassName(z), eachInputX.shape[0], np.mean(np.equal(predY, eachInputY)) * 100))
        # Loop each Validation class
        for z in range(15, 20):
            eachInputX = []
            eachInputY = []
            for k in range(vX.shape[0]):
                if vY[k] == z:
                    eachInputX.append(vX[k])
                    eachInputY.append(vY[k])
            eachInputX = np.array(eachInputX)
            eachInputY = np.array(eachInputY)
            attTmp = model.getAttribute(eachInputX)
            predY = classifier.predict(attTmp)
            print('Class: {0:<15} Size: {1:<10} Accuracy = {2:.4f}%'.format(printClassName(z), eachInputX.shape[0], np.mean(np.equal(predY, eachInputY)) * 100))
        # Loop each Test class
        for z in range(20, 32):
            eachInputX = []
            eachInputY = []
            for k in range(teX.shape[0]):
                if teY[k] == z:
                    eachInputX.append(teX[k])
                    eachInputY.append(teY[k])
            eachInputX = np.array(eachInputX)
            eachInputY = np.array(eachInputY)
            attTmp = model.getAttribute(eachInputX)
            predY = classifier.predict(attTmp)
            print('Class: {0:<15} Size: {1:<10} Accuracy = {2:.4f}%'.format(printClassName(z), eachInputX.shape[0], np.mean(np.equal(predY, eachInputY)) * 100))

        # Save sorting output
        checkX = np.concatenate((teX[:5], teX[20:25], vX[:5], vX[20:25]), axis=0)
        checkY = np.concatenate((teY[:5], teY[20:25], vY[:5], vY[20:25]), axis=0)

        a = model.getAttribute(checkX)
        b = [printClassName(x) for x in checkY]
        c = [printClassName(x) for x in classifier.predict(a)]
        d = classifier.predictScore(a)
        e = np.argsort(-d, axis=1)
        g = [[printClassName(j) for j in i] for i in e]

        tmpSupTitle = "Train Classes : "
        tmpSupTitle += " ".join(trainClass) + "\n"
        tmpSupTitle += "Validation Classes : "
        tmpSupTitle += " ".join(valClass) + "\n"
        tmpSupTitle += "Test Classes : "
        tmpSupTitle += " ".join(testClass) + "\n"
        print(tmpSupTitle)
        plt.figure(figsize=(20, 20))
        for i in range(20):
            plt.subplot(4, 5, i + 1)
            h = list(g[i])
            tmpStr = ' ' + b[i]
            for index, st in enumerate(h):
                if index % 5 == 0:
                    tmpStr += '\n'
                tmpStr += ' ' + st
            plt.title(tmpStr, multialignment='left')
            plt.imshow(checkX[i])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(globalV.FLAGS.KEY + '_Debug.png')
        plt.clf()









