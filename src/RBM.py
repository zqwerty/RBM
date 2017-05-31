#-*- coding:utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import time


def scale(X, eps=0.001):
    # scale the data points s.t the columns of the feature space
    # (i.e the predictors) are within the range [0, 1]
    # return (X - np.min(X, axis=0)) / (np.max(X, axis=0) + eps)
    # return (X - np.min(X, axis=1)) / (np.max(X, axis=1) + eps)
    return X/255


def test(X,y):
    trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.1, random_state = 42)
    logistic = LogisticRegression(C=10.0)
    logistic.fit(trainX, trainY)
    print "LOGISTIC REGRESSION ON ORIGINAL DATASET"
    print classification_report(testY, logistic.predict(testX))

    # initialize the RBM + Logistic Regression classifier with
    # the cross-validated parameters
    rbm = BernoulliRBM(n_components=200, n_iter=80,
                       learning_rate=0.001, verbose=True)
    logistic = LogisticRegression(C=100.0)

    # train the classifier and show an evaluation report
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
    classifier.fit(trainX, trainY)
    print "RBM + LOGISTIC REGRESSION ON ORIGINAL DATASET"
    print classification_report(testY, classifier.predict(testX))


def find_hyperparameter(X,y):
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.1, random_state=42)
    # perform a grid search on the 'C' parameter of Logistic
    # Regression
    print "SEARCHING LOGISTIC REGRESSION"
    params = {"C": [1.0, 10.0, 100.0]}
    start = time.time()
    gs = GridSearchCV(LogisticRegression(), params, n_jobs=-1, verbose=1)
    gs.fit(trainX, trainY)

    # print diagnostic information to the user and grab the
    # best model
    print "done in %0.3fs" % (time.time() - start)
    print "best score: %0.3f" % (gs.best_score_)
    print "LOGISTIC REGRESSION PARAMETERS"
    bestParams = gs.best_estimator_.get_params()

    # loop over the parameters and print each of them out
    # so they can be manually set
    for p in sorted(params.keys()):
        print "\t %s: %f" % (p, bestParams[p])

    # initialize the RBM + Logistic Regression pipeline
    rbm = BernoulliRBM()
    logistic = LogisticRegression()
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])

    # perform a grid search on the learning rate, number of
    # iterations, and number of components on the RBM and
    # C for Logistic Regression
    print "SEARCHING RBM + LOGISTIC REGRESSION"
    params = {
        "rbm__learning_rate": [0.1, 0.01, 0.001],
        "rbm__n_iter": [20, 40, 80],
        "rbm__n_components": [50, 100, 200],
        "logistic__C": [1.0, 10.0, 100.0]}

    # perform a grid search over the parameter
    start = time.time()
    gs = GridSearchCV(classifier, params, n_jobs=-1, verbose=1)
    gs.fit(trainX, trainY)

    # print diagnostic information to the user and grab the
    # best model
    print "\ndone in %0.3fs" % (time.time() - start)
    print "best score: %0.3f" % (gs.best_score_)
    print "RBM + LOGISTIC REGRESSION PARAMETERS"
    bestParams = gs.best_estimator_.get_params()

    # loop over the parameters and print each of them out
    # so they can be manually set
    for p in sorted(params.keys()):
        print "\t %s: %f" % (p, bestParams[p])

    # show a reminder message
    print "\nIMPORTANT"
    print "Now that your parameters have been searched, manually set"
    print "them and re-run this script with --search 0"


def test2(trainX,trainY,testX,testY):
    logistic = LogisticRegression(C=10.0)
    logistic.fit(trainX, trainY)
    print "LOGISTIC REGRESSION ON ORIGINAL DATASET"
    print classification_report(testY, logistic.predict(testX))

    # initialize the RBM + Logistic Regression classifier with
    # the cross-validated parameters
    rbm = BernoulliRBM(n_components=200, n_iter=80,
                       learning_rate=0.001, verbose=True)
    logistic = LogisticRegression(C=100.0)

    # train the classifier and show an evaluation report
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
    classifier.fit(trainX, trainY)
    print "RBM + LOGISTIC REGRESSION ON ORIGINAL DATASET"
    print classification_report(testY, classifier.predict(testX))


if __name__ == '__main__':
    np.set_printoptions(threshold=np.NaN)
    trainX = []
    trainy = []
    for i in range(4):
        f = 'data'+str(i)+'.npz'
        r = np.load(f)
        trainX.append(scale(r['X']))
        trainy.append(r['y'])
    trainX = np.concatenate((trainX[0],trainX[1],trainX[2],trainX[3]))
    trainy = np.concatenate((trainy[0], trainy[1], trainy[2], trainy[3]))
    r = np.load("test.npz")
    testX = scale(r['X'])
    testy = r['y']
    # find_hyperparameter(X, y)
    test2(trainX,trainy,testX,testy)
