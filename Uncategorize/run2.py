import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import globalV
import numpy as np
import os
import sys
from loadData import loadData
from model1 import model1
from model2 import model2
from classify import classify
from darknetModel import darknetModel
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--BASEDIR', type=str, default='/media/dataHD3/kpugdeet/', help='Base folder for dataset and logs')
    parser.add_argument('--AWA2PATH', type=str, default='AWA2/Animals_with_Attributes2/', help='Path for AWA2 dataset')
    parser.add_argument('--CUBPATH', type=str, default='CUB/CUB_200_2011/', help='Path for CUB dataset')
    parser.add_argument('--SUNPATH', type=str, default='SUN/SUNAttributeDB/', help='Path for SUN dataset')
    parser.add_argument('--APYPATH', type=str, default='APY/attribute_data/', help='Path for APY dataset')
    parser.add_argument('--GOOGLE', type=str, default='GoogleNews-vectors-negative300.bin', help='Path for google Word2Vec model')
    parser.add_argument('--KEY', type=str, default='APY',help='Choose dataset (AWA2, CUB, SUN, APY)')
    parser.add_argument('--maxSteps', type=int, default=1, help='Number of steps to run trainer.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--width', type=int, default=300, help='Width')
    parser.add_argument('--height', type=int, default=300, help='Height')
    parser.add_argument('--numClass', type=int, default=32, help='Number of class')
    parser.add_argument('--numHid', type=int, default=100, help='Number of hidden')
    parser.add_argument('--numAtt', type=int, default=64, help='Dimension of Attribute')
    parser.add_argument('--batchSize', type=int, default=32, help='Number of batch size')
    parser.add_argument('--TD', type=int, default=0, help='Train/Restore Darknet')
    parser.add_argument('--TA', type=int, default=0, help='Train/Restore Attribute')
    parser.add_argument('--TC', type=int, default=0, help='Train/Restore Classify')
    parser.add_argument('--PRE', type=int, default=4, help='1.CNN, 2.Model, 3.Classify')
    globalV.FLAGS, _ = parser.parse_known_args()

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

    # Index with total classes
    def printClassName(pos):
        if pos < len(trainClass):
            return trainClass[pos]
        elif pos < len(trainClass)+len(valClass):
            return valClass[pos-len(trainClass)]
        else:
            return testClass[pos-len(trainClass)-len(valClass)]

    # Check where there is some class that has same attributes
    print('\nCheck matching classes attributes')
    concatAtt = np.concatenate((trainAtt, valAtt, testAtt), axis=0)
    for i in range(concatAtt.shape[0]):
        for j in range(i + 1, concatAtt.shape[0]):
            if np.array_equal(concatAtt[i], concatAtt[j]):
                print('{0} {1}: {2} {3}'.format(i, printClassName(i), j, printClassName(j)))

    # Shuffle and preparing data
    divP = int(trainX.shape[0] * 0.7)
    s = np.arange(trainX.shape[0])
    np.random.shuffle(s)

    # Seen class
    trainYAtt2 = list()
    for i in range(trainY.shape[0]):
        trainYAtt2.append(trainAtt[trainY[i]])
    tmpX = trainX[s]
    tmpY = trainY[s]
    tmpYAtt = np.array(trainYAtt2)[s]

    # Unseen class
    valYAtt2 = list()
    for i in range(valY.shape[0]):
        valYAtt2.append(valAtt[valY[i]])
    valY = valY + len(trainClass)
    testYAtt2 = list()
    for i in range(testY.shape[0]):
        testYAtt2.append(testAtt[testY[i]])
    testY = testY + len(trainClass) + len(valClass)
    unseenX = np.concatenate((valX, testX), axis=0)
    unseenY = np.concatenate((valY, testY), axis=0)
    unseenYAtt = np.concatenate((valYAtt2, testYAtt2), axis=0)

    # Create all Classes
    allClassAtt = np.concatenate((np.concatenate((trainAtt, valAtt), axis=0), testAtt), axis=0)
    allClassName = np.concatenate((np.concatenate((trainClass, valClass), axis=0), testClass), axis=0)
    with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
        allClassAttName = [line.strip() for line in f]

    # Selected Seen Classes
    seen_List = [6, 8, 12]
    tmpX_S = []
    tmpY_S = []
    tmpYAtt_S = []
    for k in range(tmpY.shape[0]):
        if tmpY[k] in seen_List:
            tmpX_S.append(tmpX[k])
            tmpY_S.append(tmpY[k])
            tmpYAtt_S.append(tmpYAtt[k])
    tmpX_S = np.array(tmpX_S)
    tmpY_S = np.array(tmpY_S)
    tmpYAtt_S = np.array(tmpYAtt_S)
    print(tmpX_S.shape, tmpY_S.shape, tmpYAtt_S.shape)

    # Selected Unseen Classes
    unSeen_List = [19, 31]
    unseenX_S = []
    unseenY_S = []
    unseenYAtt_S = []
    for k in range(unseenY.shape[0]):
        if unseenY[k] in unSeen_List:
            unseenX_S.append(unseenX[k])
            unseenY_S.append(unseenY[k])
            unseenYAtt_S.append(unseenYAtt[k])
    unseenX_S = np.array(unseenX_S)
    unseenY_S = np.array(unseenY_S)
    unseenYAtt_S = np.array(unseenYAtt_S)
    print(unseenX_S.shape, unseenY_S.shape, unseenYAtt_S.shape)

    # All selected Classes
    allClassAtt_S = []
    for k in range(allClassAtt.shape[0]):
        if k in seen_List or k in unSeen_List:
            allClassAtt_S.append(allClassAtt[k])
    allClassAtt_S = np.array(allClassAtt_S)
    print(allClassAtt_S.shape)

    # Train Network
    if globalV.FLAGS.PRE == 1:
        print('\nPre-train CNN')
        darknet = darknetModel()
        darknet.trainDarknet(tmpX[:divP], tmpY[:divP], tmpX[divP:], tmpY[divP:])

    elif globalV.FLAGS.PRE == 2:
        print('\nTrain all network together')
        model = model2()
        # model.trainAtt(tmpX, tmpY, tmpYAtt, unseenX, unseenY, unseenYAtt, allClassAtt)
        model.trainAtt(tmpX_S, tmpY_S, tmpYAtt_S, unseenX_S, unseenY_S, unseenYAtt_S, allClassAtt)

    elif globalV.FLAGS.PRE == 3:
        print('\nTrain classify')
        classifier = classify()
        # classifier.trainClassify(allClassAtt, np.arange(allClassAtt.shape[0]), 0.5)
        classifier.trainClassify(allClassAtt_S, np.array([6 ,8, 12, 19, 31]), 0.5)

        # g1 = tf.Graph()
        # g2 = tf.Graph()
        # with g1.as_default():
        #     model = model2()
        # with g2.as_default():
        #     classifier = classify()
        # predictTrainAtt = model.getAttribute(tmpX)
        # duplicateUnseenAtt = np.concatenate((unseenYAtt, unseenYAtt[:3660]))
        # duplicateUnseenY = np.concatenate((unseenY, unseenY[:3660]))
        # duplicateUnseenAtt = (duplicateUnseenAtt-minAtt)/(maxAtt-minAtt)
        # trainClassAtt = np.concatenate((predictTrainAtt, duplicateUnseenAtt), axis=0)
        # trainClassY = np.concatenate((tmpY, duplicateUnseenY), axis=0)
        # print(trainClassAtt.shape, trainClassY.shape)
        # classifier.trainClassify(trainClassAtt, trainClassY)

    else:
        g1 = tf.Graph()
        g2 = tf.Graph()
        with g1.as_default():
            model = model2()
        with g2.as_default():
            classifier = classify()
        attTmp = model.getAttribute(tmpX_S)
        predY = classifier.predict(attTmp)
        print('Seen Accuracy = {0:.4f}%'.format(np.mean(np.equal(predY, tmpY_S))*100))

        attTmp = model.getAttribute(unseenX_S)
        predY = classifier.predict(attTmp)
        print('Unseen Accuracy = {0:.4f}%'.format(np.mean(np.equal(predY, unseenY_S)) * 100))

    # 31 Horse, 30 Bicycle, 28 Cat, 22 Car, 16 Bird, 11 Dog
    tmpNameFile = 'Horse'
    # horseInput = []
    # for k in range(unseenY.shape[0]):
    #     if unseenY[k] == 31:
    #         horseInput.append(unseenX[k])
    # horseInput = np.array(horseInput)
    s = np.arange(unseenX_S.shape[0])
    np.random.shuffle(s)
    horseInput = unseenX_S[s]

    g1 = tf.Graph()
    g2 = tf.Graph()
    with g1.as_default():
        model = model2()
    with g2.as_default():
        classifier = classify()
    a = model.getAttribute(horseInput[:10])
    # b = [printClassName(x) for x in  model.predict(horseInput[:10], allClassAtt)]
    b = [printClassName(x) for x in classifier.predict(a)]
    print(b)

    import plotly.plotly as py
    import plotly.graph_objs as go

    py.sign_in('krittaphat.pug', 'oVTAbkhd2RQvodGOwrwp')
    allClassNormalize = np.round(allClassAtt)
    trace = go.Heatmap(z = allClassNormalize,
                       y = allClassName,
                       x = allClassAttName)
    data = [trace]
    layout = go.Layout(title=globalV.FLAGS.KEY, width=1920, height=1080)
    fig = go.Figure(data=data, layout=layout)
    py.image.save_as(fig, filename='Z_'+globalV.FLAGS.KEY+'_Heat.png')

    trace = go.Heatmap(z = a,
                       x = allClassAttName,
                       y = b)
    data = [trace]
    layout = go.Layout(title=globalV.FLAGS.KEY+'_'+tmpNameFile, width=1920, height=1080,
                       yaxis=dict(
                           ticktext = b,
                           tickvals = np.arange(len(b))))
    fig = go.Figure(data=data, layout=layout)
    py.image.save_as(fig, filename=globalV.FLAGS.KEY + '_Att_' + tmpNameFile + '.png')

    print('\nCheck predict output')
    # predictLabel = model.predict(horseInput[:10], allClassAtt)
    predictLabel = classifier.predict(a)
    print(predictLabel)
    print('Save image to '+globalV.FLAGS.KEY+'_'+tmpNameFile+'.png')
    plt.figure(figsize=(6, 24))
    for i in range(10):
        plt.subplot(10, 2, i+1)
        plt.title(printClassName(predictLabel[i]))
        plt.imshow(horseInput[i], aspect='auto')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(globalV.FLAGS.KEY+'_'+tmpNameFile+'.png')
    plt.clf()









