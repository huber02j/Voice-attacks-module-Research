#!/usr/bin/env python3

import numpy as np
import scipy.io.wavfile as wav
from os import listdir
from python_speech_features import mfcc, delta, logfbank
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys
from datetime import datetime
from sklearn.metrics import confusion_matrix
import random
from imblearn.over_sampling import SMOTE

def main():
    train = 0
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        while args:
            modelName = args.pop()
    else:
        train = 1

    if train:
        x,y = getData()
        C=[0.01, 0.1, 1, 10]
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        n_estimators = [1,10,100]
        max_depth = [1,10,100,None]
        trials = len(C)*len(kernel)+len(n_estimators)*len(max_depth)
        trial = 1
        results = []
        for cvalue in C:
            for k in kernel:
                score,name = train_model('SVC',x,y,C=cvalue, kernel=k) # initialize machine learning model
                results.append((score,name))
                print("Loading: |{}{}|".format('▓'*trial,'_'*(trials-trial)),end="\r")
                trial += 1
        for n in n_estimators:
            for d in max_depth:
                score,name = train_model('RandomForestClassifier',x,y, n_estimators=n, max_depth=d)
                results.append((score,name))
                print("Loading: |{}{}|".format('▓'*trial,'_'*(trials-trial)),end="\r")
                trial += 1


        length = max([len(name) for score,name in results])
        results = sorted(results,reverse=True)
        for score, name in results:
            print("{:<{}}{}".format(name, str(length+3), score))
        clf = load_model(results[0][1])
        print("Test\n{}\nAccuracy: {}".format(results[0][1],str(test_model(clf,"FinalProjectTestRecordingsALL"))))

    else:
        clf = load_model(modelName)
        print("Accuracy: " + str(test_model(clf)))


def getSlash():
    if len(sys.platform) >= 3 and sys.platform[0:3] == "win":
        slash = "\\"
    else:
        slach = "/"

    return slash

def getInputs(fileNames, dir):
    x =[]
    y = []
    count = 0

    slash = getSlash()
    if dir == "FinalProjectTrainRecordings":
        dataFile = open("data.csv",'w+')
        dataFile.write(','.join([','.join([x+str(i) for x in ["avg","max","min"]]) for i in range(1,14)]) + ",class\n")
    for file in fileNames:
        count += 1

        recType = file.split(slash)[0][0] # G or R
        (rate,sig) = wav.read(dir + slash + file) #This slash might need to be a forward slash for a different machine
        sig = sig[:,0]
        mfcc_feat = mfcc(sig,rate)
        # d_mfcc_feat = delta(mfcc_feat, 2)
        # fbank_feat = logfbank(sig,rate)
        calc_info = calc(mfcc_feat)
        x.append(calc_info)
        recTypeInt = int(recType == 'G') #Genuine is 1, Replayed is 0
        y.append(recTypeInt)
        if dir == "FinalProjectTrainRecordings":
            dataFile.write(','.join([str(x) for x in calc_info]) + ',' + str(recType) + '\n')
    if dir == "FinalProjectTrainRecordings":
        dataFile.close()

    x = np.array(x)
    y = np.array(y)

    return (x,y)

def calc(mfcc):
    final = []

    for i in range(len(mfcc[0])):
        column = [x[i] for x in mfcc]

        final.append(sum(column)/len(column))
        final.append(max(column))
        final.append(min(column))
    return final

def getData():
    dir = "FinalProjectTrainRecordings"
    trainFolders = listdir(dir)
    trainFiles = []
    slash = getSlash()

    Random = 1
    smote = 0

    maxFiles = max([len(listdir(dir + slash + folder)) for folder in trainFolders])

    for folder in trainFolders:
        path = dir + slash + folder
        #N = min(5, len(listdir(path))/2)
        names = list(filter(lambda x: x[0] != '.' and x.split(".")[-1] == "wav", listdir(path)))


        if len(names) < maxFiles:
            if Random:
                namesLength = len(names)
                for _ in range(maxFiles-namesLength):
                    names.append(listdir(path)[random.randint(0,len(listdir(path))-1)])

        #names = names[:N] + names[-N:]
        trainFiles += list(map(lambda x: "{}{}{}".format(folder,slash,x), names))

    x,y = getInputs(trainFiles, dir)
    if smote:
        sm = SMOTE(random_state=42)
        x_res, y_res = sm.fit_resample(x, y)
        return x_res,y_res

    return x,y

def train_model(model,x,y,C=1, kernel="rbf", n_estimators=100, max_depth=2):
    # for Training
    if model == "SVC":
        clf = SVC(C=C, kernel=kernel, gamma='auto') # initialize machine learning model
    else:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    # max_features
    clf.fit(x, y)
    score = test_model(clf)
    #save model
    time = ".".join("_".join(str(datetime.now()).split()).split(":"))
    if model == 'SVC':
        savedModelName = "model_{}_{}_{}.sav".format(model,C,kernel)
    else:
        if max_depth:
            savedModelName = "model_{}_{}_{}.sav".format(model,n_estimators,max_depth)
        else:
            savedModelName = "model_{}_{}_None.sav".format(model,n_estimators)

    # print("{}: {}".format(savedModelName,score))

    pickle.dump(clf, open(savedModelName, 'wb'))
    # print("Saved model as: " + savedModelName)

    return (score, savedModelName)


def load_model(filename):
    clf = pickle.load(open(filename, 'rb'))
    return clf

def test_model(clf,dir="FinalProjectValRecordings"):
    testFolders = listdir(dir)
    testFiles = []
    slash = getSlash()

    maxFiles = max([len(listdir(dir + slash + folder)) for folder in testFolders])

    for folder in testFolders:
        path = dir + slash + folder
        #N = min(1, len(listdir(path))/2)
        names = list(filter(lambda x: x[0] != '.' and x.split(".")[-1] == "wav", listdir(path)))
        #names = names[:N] + names[-N:]
        # if len(names) < maxFiles:
        #     namesLength = len(names)
        #     for _ in range(maxFiles-namesLength):
        #         names.append(listdir(path)[random.randint(0,len(listdir(path))-1)])

        testFiles += list(map(lambda x: "{}{}{}".format(folder,slash,x), names))


    x,actual = getInputs(testFiles, dir)

    #unseen_X = np.array[ [1,2,3,...., 13] ];

    if dir == "FinalProjectTestRecordingsALL":
        predicted = clf.predict(x) #guess
        print(predicted)
        print(actual)
        print(confusion_matrix(predicted,actual))


    return clf.score(x,actual)


if __name__ == '__main__':
    main()
