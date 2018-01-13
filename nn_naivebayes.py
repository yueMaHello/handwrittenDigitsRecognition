import data_loader
from sklearn.cluster import KMeans
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import utilities as utils


class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params, parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest


class NaiveBayes(Classifier):
    def __init__(self, parameters={}):
        self.params = {'cv': None}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.numcv = self.params['cv']

    def cv(self, Xtrain, ytrain):
        ytrain = np.array(ytrain, ndmin=2)
        if self.numcv is not None:
            c = self.numcv
        else:
            return {0: Xtrain}, {0: ytrain}
        l = len(ytrain)
        xresult = {}
        yresult = {}
        for i in range(c):
            xresult[i] = Xtrain[i * l / c:((i + 1) * l / c)]
            yresult[i] = ytrain[i * l / c:((i + 1) * l / c)]

        return xresult, yresult

    def learn(self, Xtrain, ytrain):
        x, y = self.cv(Xtrain, ytrain)
        selection = [BernoulliNB(), MultinomialNB(), GaussianNB()]
        err = []
        for i in selection:
            e = 0
            for j in x.keys():
                clf = i
                Xtestset = []
                Ytestset = []
                for p in x.keys():
                    if p == j:
                        continue
                    else:
                        Xtestset.append(x[p])
                        Ytestset.append(y[p])
                xt = Xtestset[0]
                yt = Ytestset[0]
                for t in range(1, len(Xtestset)):
                    xt = np.append(xt, Xtestset[t], axis=0)
                    yt = np.append(yt, Ytestset[t], axis=0)

                yd = y_digit(yt)
                clf.fit(xt, yd)
                ytd = y_digit(y[j])
                e = e + hamming(clf.predict(x[j]), ytd)
            err.append(e)
            print e
        err = np.array(err)
        s = np.where(err[err == np.min(err)])[0][0]
        self.clf = selection[s]
        Ytrain = y_digit(ytrain)
        self.clf.fit(Xtrain, Ytrain)

    def predict(self, Xtest):
        return self.clf.predict(Xtest)


class NeuralNetwork(Classifier):
    def __init__(self, parameters={}):
        self.params = {'cv': None}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.numcv = self.params['cv']

    def cv(self, Xtrain, ytrain):
        ytrain = np.array(ytrain, ndmin=2)
        if self.numcv is not None:
            c = self.numcv
        else:
            return {0: Xtrain}, {0: ytrain}
        l = len(ytrain)
        xresult = {}
        yresult = {}
        for i in range(c):
            xresult[i] = Xtrain[i * l / c:((i + 1) * l / c)]
            yresult[i] = ytrain[i * l / c:((i + 1) * l / c)]

        return xresult, yresult

    def learn(self, Xtrain, ytrain):
        x, y = self.cv(Xtrain, ytrain)
        selection = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        err = []
        '''for i in selection:
            e=0
            for j in x.keys():
                clf=MLPClassifier(solver='sgd', alpha=0,max_iter=1000,activation='relu',
                    hidden_layer_sizes=(500, i), random_state=1)

                Xtestset=[]
                Ytestset=[]
                for p in x.keys():
                    if p==j:
                        continue
                    else:
                        Xtestset.append(x[p])
                        Ytestset.append(y[p])
                xt=Xtestset[0]
                yt=Ytestset[0]
                for t in range(1,len(Xtestset)):
                    xt=np.append(xt,Xtestset[t],axis=0)
                    yt=np.append(yt,Ytestset[t],axis=0)


                yd=y_digit(yt)  
                clf.fit(xt,yd)
                ytd=y_digit(y[j])
                e=e+hamming(clf.predict(x[j]),ytd)
            err.append(e)
            print e
            print i
        err=np.array(err)
        s=np.where(err[err==np.min(err)])[0][0]'''
        self.clf = MLPClassifier(solver='sgd', alpha=0, max_iter=1500, activation='relu',
                                 hidden_layer_sizes=(500, 10), tol=1e-6, random_state=1)

        Ytrain = y_digit(ytrain)
        self.clf.fit(Xtrain, Ytrain)
        # print selection[s]

    def predict(self, Xtest):
        return self.clf.predict(Xtest)


def y_digit(ytrain):
    k = np.zeros(ytrain.shape[0], dtype=np.int)
    for i in range(ytrain.shape[0]):
        if len(np.where(ytrain[i] == 1)[0]) != 0:
            k[i] = int(np.where(ytrain[i] == 1)[0][0])
    return k


def hamming(s1, s2):
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


def getaccuracy(ytest, predictions):
    correct = 0

    return (1 - hamming(ytest, predictions) / float(len(ytest))) * 100.0


if __name__ == '__main__':
    trainset, testset = data_loader.load_train_and_test_data(1000, 500)
    Xtrain = np.array(trainset[0], ndmin=2)
    Xtest = np.array(testset[0], ndmin=2)
    ytest = testset[1]

    ytrain = trainset[1]
    y = y_digit(ytrain)
    yt = y_digit(ytest)
    clf = MLPClassifier(solver='sgd', alpha=1e-6, max_iter=1500, activation='identity', hidden_layer_sizes=(500, 10),
                        random_state=1)
    clf.fit(Xtrain, y)
    naive_bayes = BernoulliNB()
    naive_bayes.fit(Xtrain, y)
    print getaccuracy(clf.predict(Xtest), yt)
