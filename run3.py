import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import globalV
import numpy as np
import os
import sys
from loadData import loadData
from attribute import attribute
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
    parser.add_argument('--numAtt', type=int, default=320, help='Dimension of Attribute')
    parser.add_argument('--batchSize', type=int, default=32, help='Number of batch size')
    parser.add_argument('--TD', type=int, default=0, help='Train/Restore Darknet')
    parser.add_argument('--TA', type=int, default=0, help='Train/Restore Attribute')
    parser.add_argument('--TC', type=int, default=0, help='Train/Restore Classify')
    parser.add_argument('--PRE', type=int, default=5, help='1.CNN, 2.Model, 3.Classify')
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

    concatAtt = np.concatenate((trainAtt, valAtt, testAtt), axis=0)

    # Quantization
    bins = np.array([-1, 0.2, 0.4, 0.6, 0.8])
    tmp_Q = np.digitize(concatAtt, bins, right=True)
    tmp_Q = tmp_Q - 1
    concatAtt_D = np.zeros((tmp_Q.shape[0], tmp_Q.shape[1], 5))
    for i in range(tmp_Q.shape[0]):
        concatAtt_D[i][np.arange(tmp_Q.shape[1]), tmp_Q[i]] = 1
    concatAtt_D = concatAtt_D.reshape(tmp_Q.shape[0], -1)
    print('\nNew attribute shape {0}'.format(concatAtt_D.shape))

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

    # Create all Classes
    allClassName = np.concatenate((np.concatenate((trainClass, valClass), axis=0), testClass), axis=0)
    with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
        allClassAttName = [line.strip() for line in f]

    # Train Network
    if globalV.FLAGS.PRE == 1:
        print('\nPre-train CNN')
        darknet = darknetModel()
        darknet.trainDarknet(trX[:trDiv], trY[:trDiv], trX[trDiv:], trY[trDiv:])

    elif globalV.FLAGS.PRE == 2:
        print('\nTrain all network together')
        attModel = attribute()
        attModel.trainAtt(trX, trAtt, vX, vAtt)

    elif globalV.FLAGS.PRE == 3:
        print('\nTrain classify')
        # classifier = classify()
        # classifier.trainClassify(concatAtt_D, np.arange(concatAtt_D.shape[0]), 0.5)

        g1 = tf.Graph()
        g2 = tf.Graph()
        with g1.as_default():
            model = attribute()
        with g2.as_default():
            classifier = classify()
        predictTrainAtt = model.getAttribute(trX)
        combineAtt = np.concatenate((concatAtt_D, predictTrainAtt, vAtt, teAtt), axis=0)
        combineY = np.concatenate((np.arange(concatAtt_D.shape[0]), trY, valY, teY), axis=0)
        print(combineAtt.shape, combineY.shape)
        classifier.trainClassify(combineAtt, combineY, 0.5)

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

        # Check Class
        import plotly.plotly as py
        import plotly.graph_objs as go

        b = [printClassName(x) for x in np.arange(concatAtt_D.shape[0])]
        py.sign_in('krittaphat.pug', 'oVTAbkhd2RQvodGOwrwp')
        trace = go.Heatmap(z=concatAtt_D,
                           y=b)
        data = [trace]
        layout = go.Layout(title=globalV.FLAGS.KEY, width=1920, height=1080,
                           yaxis=dict(
                               ticktext=b,
                               tickvals=np.arange(len(b))))
        fig = go.Figure(data=data, layout=layout)
        py.image.save_as(fig, filename='Z_' + globalV.FLAGS.KEY + '_Heat.png')

    # Accuracy for each class
    print('')
    g1 = tf.Graph()
    g2 = tf.Graph()
    with g1.as_default():
        model = attribute()
    with g2.as_default():
        classifier = classify()
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

    # # Debug
    # checkX = teX[:5]
    # checkY = teY[:5]
    # g1 = tf.Graph()
    # g2 = tf.Graph()
    # with g1.as_default():
    #     model = attribute()
    # with g2.as_default():
    #     classifier = classify()
    #
    # a = model.getAttribute(checkX)
    # b = [printClassName(x) for x in checkY]
    # c = [printClassName(x) for x in classifier.predict(a)]
    # d = classifier.predictScore(a)
    # e = np.argsort(-d, axis=1)
    # f = e[:,:5]
    # g = [[printClassName(j) for j in i] for i in f]
    #
    # plt.figure(figsize=(5, 20))
    # for i in range(5):
    #     plt.subplot(5, 1, i+1)
    #     h = list(g[i])
    #     h.append(b[i])
    #     plt.title(h)
    #     plt.imshow(checkX[i], aspect='auto')
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('Z_' + globalV.FLAGS.KEY + '_Debug.png')
    # plt.clf()





    # # Check Output
    # checkX = vX[:50]
    # checkY = vY[:50]
    # g1 = tf.Graph()
    # g2 = tf.Graph()
    # with g1.as_default():
    #     model = attribute()
    # a = model.getAttribute(checkX)
    # b = [printClassName(x) for x in checkY]
    #
    # import plotly.plotly as py
    # import plotly.graph_objs as go
    #
    # py.sign_in('krittaphat.pug', 'oVTAbkhd2RQvodGOwrwp')
    # trace = go.Heatmap(z=a,
    #                    y=b)
    # data = [trace]
    # layout = go.Layout(title=globalV.FLAGS.KEY, width=1920, height=1080,
    #                    yaxis=dict(
    #                        ticktext=b,
    #                        tickvals=np.arange(len(b))))
    # fig = go.Figure(data=data, layout=layout)
    # py.image.save_as(fig, filename='Z_' + globalV.FLAGS.KEY + '_Heat.png')

    # # 31 Horse, 30 Bicycle, 28 Cat, 22 Car, 16 Bird, 11 Dog
    # tmpNameFile = 'Horse'
    # # horseInput = []
    # # for k in range(unseenY.shape[0]):
    # #     if unseenY[k] == 31:
    # #         horseInput.append(unseenX[k])
    # # horseInput = np.array(horseInput)
    # s = np.arange(unseenX_S.shape[0])
    # np.random.shuffle(s)
    # horseInput = unseenX_S[s]
    #
    # g1 = tf.Graph()
    # g2 = tf.Graph()
    # with g1.as_default():
    #     model = model2()
    # with g2.as_default():
    #     classifier = classify()
    # a = model.getAttribute(horseInput[:10])
    # # b = [printClassName(x) for x in  model.predict(horseInput[:10], allClassAtt)]
    # b = [printClassName(x) for x in classifier.predict(a)]
    # print(b)
    #
    # import plotly.plotly as py
    # import plotly.graph_objs as go
    #
    # py.sign_in('krittaphat.pug', 'oVTAbkhd2RQvodGOwrwp')
    # trace = go.Heatmap(z = concatAtt_D,
    #                    y = allClassName)
    #                    # x = allClassAttName)
    # data = [trace]
    # layout = go.Layout(title=globalV.FLAGS.KEY, width=1920, height=1080)
    # fig = go.Figure(data=data, layout=layout)
    # py.image.save_as(fig, filename='Z_'+globalV.FLAGS.KEY+'_Heat.png')
    #
    # trace = go.Heatmap(z = a,
    #                    x = allClassAttName,
    #                    y = b)
    # data = [trace]
    # layout = go.Layout(title=globalV.FLAGS.KEY+'_'+tmpNameFile, width=1920, height=1080,
    #                    yaxis=dict(
    #                        ticktext = b,
    #                        tickvals = np.arange(len(b))))
    # fig = go.Figure(data=data, layout=layout)
    # py.image.save_as(fig, filename=globalV.FLAGS.KEY + '_Att_' + tmpNameFile + '.png')
    #
    # print('\nCheck predict output')
    # # predictLabel = model.predict(horseInput[:10], allClassAtt)
    # predictLabel = classifier.predict(a)
    # print(predictLabel)
    # print('Save image to '+globalV.FLAGS.KEY+'_'+tmpNameFile+'.png')
    # plt.figure(figsize=(6, 24))
    # for i in range(10):
    #     plt.subplot(10, 2, i+1)
    #     plt.title(printClassName(predictLabel[i]))
    #     plt.imshow(horseInput[i], aspect='auto')
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(globalV.FLAGS.KEY+'_'+tmpNameFile+'.png')
    # plt.clf()









