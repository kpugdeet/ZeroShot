import os
import gensim
import pickle
import numpy as np
import globalV
import glob
import cv2

class loadData(object):
    @staticmethod
    def resize(imageName):
        img = cv2.imread(imageName)
        h_img, w_img, _ = img.shape
        imgResize = cv2.resize(img, (globalV.FLAGS.width, globalV.FLAGS.height))
        imgRGB = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
        imgResizeNp = np.asarray(imgRGB)
        #imgResizeNp = (imgResizeNp / 255.0) * 2.0 - 1.0
        return imgResizeNp

    # Read AwA2 dataset
    @staticmethod
    def readAwA2():
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + 'trainclasses.txt', 'r') as f:
            trainClass = [line.strip() for line in f]
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + 'testclasses.txt', 'r') as f:
            testClass = [line.strip() for line in f]
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + 'classes.txt', 'r') as f:
            allClass = [line.split()[1] for line in f]
        valClass = trainClass[27:]
        trainClass = trainClass[:27]

        # Read attribute value for each class
        trainAtt = list()
        testAtt = list()
        valAtt = list()
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + 'predicate-matrix-binary.txt', 'r') as f:
            for _, line in enumerate(f):
                if allClass[_] in trainClass:
                    trainAtt.append(line.split())
                elif allClass[_] in valClass:
                    valAtt.append(line.split())
                else:
                    testAtt.append(line.split())
        trainAtt = np.array(trainAtt).astype(np.float64)
        testAtt = np.array(testAtt).astype(np.float64)
        valAtt = np.array(testAtt).astype(np.float64)

        # Read train image
        trainX = list()
        trainY = list()
        for _, name in enumerate(trainClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + "JPEGImages/" + name + "/*.jpg"):
                trainX.append(loadData.resize(image))
                trainY.append(_)
        trainX = np.array(trainX).astype(np.uint8)
        trainY = np.array(trainY).astype(np.uint8)

        # Read valid image
        valX = list()
        valY = list()
        for _, name in enumerate(valClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + "JPEGImages/" + name + "/*.jpg"):
                valX.append(loadData.resize(image))
                valY.append(_)
        valX = np.array(valX).astype(np.uint8)
        valY = np.array(valY).astype(np.uint8)

        # Read test image
        testX = list()
        testY = list()
        for _, name in enumerate(testClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + "JPEGImages/" + name + "/*.jpg"):
                testX.append(loadData.resize(image))
                testY.append(_)
        testX = np.array(testX).astype(np.uint8)
        testY = np.array(testY).astype(np.uint8)
        return (trainClass, trainAtt, trainX, trainY), (valClass, valAtt, valX, valY), (testClass, testAtt, testX, testY)

    # Read AwA2 dataset
    @staticmethod
    def readCUB():
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + 'trainclasses.txt', 'r') as f:
            trainClass = [line.split()[1] for line in f]
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + 'testclasses.txt', 'r') as f:
            testClass = [line.split()[1] for line in f]
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + 'classes.txt', 'r') as f:
            allClass = [line.split()[1] for line in f]
        valClass = trainClass[100:]
        trainClass = trainClass[:100]

        # Read attribute value for each class
        trainAtt = list()
        testAtt = list()
        valAtt = list()
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + 'attributes/class_attribute_labels_continuous.txt', 'r') as f:
            for _, line in enumerate(f):
                if allClass[_] in trainClass:
                    trainAtt.append(line.split())
                elif allClass[_] in valClass:
                    valAtt.append(line.split())
                else:
                    testAtt.append(line.split())
        trainAtt = np.array(trainAtt).astype(np.float64)
        testAtt = np.array(testAtt).astype(np.float64)
        valAtt = np.array(testAtt).astype(np.float64)

        # Read train image
        trainX = list()
        trainY = list()
        for _, name in enumerate(trainClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + "images/" + name + "/*.jpg"):
                trainX.append(loadData.resize(image))
                trainY.append(_)
        trainX = np.array(trainX).astype(np.uint8)
        trainY = np.array(trainY).astype(np.uint8)

        # Read valid image
        valX = list()
        valY = list()
        for _, name in enumerate(valClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + "images/" + name + "/*.jpg"):
                valX.append(loadData.resize(image))
                valY.append(_)
        valX = np.array(valX).astype(np.uint8)
        valY = np.array(valY).astype(np.uint8)

        # Read test image
        testX = list()
        testY = list()
        for _, name in enumerate(testClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + "images/" + name + "/*.jpg"):
                testX.append(loadData.resize(image))
                testY.append(_)
        testX = np.array(testX).astype(np.uint8)
        testY = np.array(testY).astype(np.uint8)

        # Change class name (remove number)
        trainClass = [x.split('.')[1] for x in trainClass]
        valClass = [x.split('.')[1] for x in valClass]
        testClass = [x.split('.')[1] for x in testClass]

        return (trainClass, trainAtt, trainX, trainY), (valClass, valAtt, valX, valY), (testClass, testAtt, testX, testY)

    # Generate vector of word
    @staticmethod
    def getWord2Vec(inputName):
        vector = list()
        model = gensim.models.KeyedVectors.load_word2vec_format(globalV.FLAGS.BASEDIR + globalV.FLAGS.GOOGLE, binary=True)
        inputName = [x.replace('+', '_') for x in inputName]
        for word in inputName:
            try:
                vector.append(model[word])
            except KeyError as e:
                print(e)
                tmp = word.split('_')
                tmpModel = model[tmp[0]]
                for x in tmp[1:]:
                    tmpModel += model[x]
                vector.append(tmpModel)
                continue
        return np.array(vector).astype(np.float64)

    @staticmethod
    def getData():
        # Get data
        if os.path.isfile(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/trainClass.pkl'):
            trainClass = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/trainClass.pkl', 'rb'))
            trainAtt = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/trainAtt.pkl', 'rb'))
            trainVec = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/trainVec.pkl', 'rb'))
            trainX = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/trainX.pkl.npy')
            trainY = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/trainY.pkl.npy')

            valClass = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/valClass.pkl', 'rb'))
            valAtt = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/valAtt.pkl', 'rb'))
            valVec = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/valVec.pkl', 'rb'))
            valX = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/valX.pkl.npy')
            valY = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/valY.pkl.npy')

            testClass = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/testClass.pkl', 'rb'))
            testAtt = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/testAtt.pkl', 'rb'))
            testVec = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/testVec.pkl', 'rb'))
            testX = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/testX.pkl.npy')
            testY = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/testY.pkl.npy')
        else:
            if globalV.FLAGS.KEY == 'AWA2':
                (trainClass, trainAtt, trainX, trainY), (valClass, valAtt, valX, valY), (testClass, testAtt, testX, testY) = loadData.readAwA2()
            if globalV.FLAGS.KEY == 'CUB':
                (trainClass, trainAtt, trainX, trainY), (valClass, valAtt, valX, valY), (testClass, testAtt, testX, testY) = loadData.readCUB()
            trainVec = loadData.getWord2Vec(trainClass)
            testVec = loadData.getWord2Vec(testClass)
            valVec = loadData.getWord2Vec(valClass)

            pickle.dump(trainClass, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/trainClass.pkl', 'wb'))
            pickle.dump(trainAtt, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/trainAtt.pkl', 'wb'))
            pickle.dump(trainVec, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/trainVec.pkl', 'wb'))
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/trainX.pkl', trainX)
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/trainY.pkl', trainY)

            pickle.dump(valClass, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/valClass.pkl', 'wb'))
            pickle.dump(valAtt, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/valAtt.pkl', 'wb'))
            pickle.dump(valVec, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/valVec.pkl', 'wb'))
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/valX.pkl', valX)
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/valY.pkl', valY)

            pickle.dump(testClass, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/testClass.pkl', 'wb'))
            pickle.dump(testAtt, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/testAtt.pkl', 'wb'))
            pickle.dump(testVec, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/testVec.pkl', 'wb'))
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/testX.pkl', testX)
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.KEY + '/backup/testY.pkl', testY)
        return (trainClass, trainAtt, trainVec, trainX, trainY), (valClass, valAtt, valVec, valX, valY), (testClass, testAtt, testVec, testX, testY)
