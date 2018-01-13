from __future__ import division  # floating point division
import numpy as np
import utilities as utils

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

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


class LogitReg(Classifier):
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)


    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))


    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """
        cost=0.0
        ### YOUR CODE HERE
        yhat =utils.sigmoid(np.dot(X, theta.T))
        cost =-np.dot(y,np.log(yhat))-np.dot((1-y),np.log(1-yhat))+self.params['regwgt']*self.regularizer[0](theta)
        ###END YOUR CODE
        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """
        grad = np.zeros(len(theta))
        ### YOUR CODE HERE
        grad=np.dot((utils.sigmoid(np.dot(X, theta.T)) - y).T,X)+self.params['regwgt']*self.regularizer[1](theta)
        #ask ta
        return grad

    def learn(self, Xtrain, ytrain,stepsize):
        """
        Learn the weights using the training data
        """
        self.weights = np.zeros(Xtrain.shape[1], )
        ### YOUR CODE HERE
        epoch =1500
        w = np.zeros((ytrain.shape[1],Xtrain.shape[1]))

        for i in range(epoch):
            Xtrain, ytrain = self.unison_shuffled_copies(Xtrain, ytrain)
            for j in range(Xtrain.shape[0]):
                X = np.array(Xtrain[j, :], ndmin=2)
                y = np.array(ytrain[j,:],ndmin = 2)
                g= self.logit_cost_grad(w,X,y)
                w = w - (stepsize * 1.0/(i + 1))*g
        self.weights = w
        return w
        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ### YOUR CODE HERE
        value = utils.sigmoid(np.dot(Xtest, self.weights.T))
        ytest = np.zeros(value.shape)
        for i in range(value.shape[0]):
            maxIndex = 0
            maxValue = 0
            for j in range(value.shape[1]):
                if value[i][j]>maxValue:
                    maxIndex = j
                    maxValue = value[i][j]
            ytest[i][maxIndex] = 1
        ytest = self.y_digit(ytest)
        print 6666

        ### END YOUR CODE
        assert len(ytest) == Xtest.shape[0]
        return ytest
    def unison_shuffled_copies(self, x1, x2):
        randomize = np.arange(len(x1))
        np.random.shuffle(randomize)
        return x1[randomize],x2[randomize]

    def y_digit(self,ytrain):
        k = np.zeros(ytrain.shape[0])
        for i in range(ytrain.shape[0]):
            if len(np.where(ytrain[i] == 1)[0]) != 0:
                k[i] = np.where(ytrain[i] == 1)[0][0]
        return k

    def hamdist(self,str1, str2):
        """Count the # of differences between equal length strings str1 and str2"""
        diffs = 0
        for ch1, ch2 in zip(str1, str2):
            if ch1 != ch2:
                diffs += 1
        return diffs

    def getaccuracy(self,ytest, predictions):
        return (1 - self.hamdist(ytest, predictions) / float(len(ytest))) * 100.0

    def cross_validation_learn(self,Xtrain,ytrain):
        stepsizeList =[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
        errorList =[0]*10
        bestErrorIndex= 0
        bestError = 1000000000
        length = Xtrain.shape[0]
        XtrainList = [Xtrain[0:250],Xtrain[250:500],Xtrain[500:750],Xtrain[750:1000]]

        ytrainList = [ytrain[0:250],ytrain[250:500],ytrain[500:750],ytrain[750:1000]]

        for i in range(len(stepsizeList)):
            stepsize = stepsizeList[i]

            newXtrain = np.concatenate((XtrainList[0], XtrainList[1],XtrainList[2]), axis=0)
            newYtrain = np.concatenate((ytrainList[0],ytrainList[1],ytrainList[2]),axis=0)
            self.learn(newXtrain,newYtrain,stepsize)#first learn
            predictions1 = self.predict(XtrainList[3])
            ytestset = self.y_digit(ytrainList[3])
            error1 = self.hamdist(ytestset,predictions1)

            newXtrain = np.concatenate((XtrainList[0], XtrainList[1],XtrainList[3]), axis=0)
            newYtrain = np.concatenate((ytrainList[0],ytrainList[1],ytrainList[3]),axis=0)
            self.learn(newXtrain,newYtrain,stepsize)#first learn
            predictions2 = self.predict(XtrainList[2])
            ytestset = self.y_digit(ytrainList[2])
            error2 = self.hamdist(ytestset,predictions2)


            newXtrain = np.concatenate((XtrainList[0], XtrainList[2],XtrainList[3]), axis=0)
            newYtrain = np.concatenate((ytrainList[0],ytrainList[2],ytrainList[3]),axis=0)
            self.learn(newXtrain,newYtrain,stepsize)#first learn
            predictions3 = self.predict(XtrainList[1])
            ytestset = self.y_digit(ytrainList[1])
            error3 = self.hamdist(ytestset,predictions3)


            newXtrain = np.concatenate((XtrainList[1], XtrainList[2],XtrainList[3]), axis=0)
            newYtrain = np.concatenate((ytrainList[1],ytrainList[2],ytrainList[3]),axis=0)
            self.learn(newXtrain,newYtrain,stepsize)#first learn
            predictions4 = self.predict(XtrainList[0])
            ytestset = self.y_digit(ytrainList[0])
            error4 = self.hamdist(ytestset,predictions4)

            print error1+error2+error3+error4
            if (error1+error2+error3+error4)/4<bestError:
                bestError = (error1+error2+error3+error4)/4
                bestErrorIndex = i

        bestStepsize = stepsizeList[bestErrorIndex]
        self.weights = self.learn(Xtrain,ytrain,bestStepsize)
        return bestStepsize,self.weights

