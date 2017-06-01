#-*- coding:utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model, datasets, metrics
from scipy.ndimage import convolve
import time


def scale(X, eps=0.001):
    # scale the data points s.t the columns of the feature space
    # (i.e the predictors) are within the range [0, 1]
    # return (X - np.min(X, axis=0)) / (np.max(X, axis=0) + eps)
    # return (X - np.min(X, axis=1)) / (np.max(X, axis=1) + eps)
    return X/255


def logistic_regression(trainX, trainY):
    logistic = LogisticRegression(C=1.0)
    logistic.fit(trainX, trainY)
    return logistic


def rbm_lr(trainX, trainY):
    rbm = BernoulliRBM(n_components=200, n_iter=20,
                       learning_rate=0.001, verbose=True)
    logistic = LogisticRegression(C=10000.0)

    # train the classifier and show an evaluation report
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
    classifier.fit(trainX, trainY)
    return classifier


def rbm2_lr(trainX,trainY):
    rbm1 = BernoulliRBM(n_components=500, n_iter=80,
                       learning_rate=0.001, verbose=True)
    rbm2 = BernoulliRBM(n_components=200, n_iter=80,
                       learning_rate=0.001, verbose=True)
    logistic = LogisticRegression(C=100.0)

    # train the classifier and show an evaluation report
    classifier = Pipeline([("rbm1", rbm1), ("rbm2", rbm2), ("logistic", logistic)])
    classifier.fit(trainX, trainY)
    return classifier


def test_model(testX, testY, model):
    print classification_report(testY, model.predict(testX))


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
        "rbm__learning_rate": [0.001],
        "rbm__n_iter": [80],
        "rbm__n_components": [200],
        "logistic__C": [100.0,1000,10000]}

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


def test_mnist():
    # Load Data
    digits = datasets.load_digits()
    X = np.asarray(digits.data, 'float32')
    X, Y = nudge_dataset(X, digits.target)
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)
    test_model(X_test,Y_test, logistic_regression(X_train, Y_train))
    test_model(X_test,Y_test, rbm_lr(X_train,Y_train))
    # # Models we will use
    # logistic = linear_model.LogisticRegression()
    # rbm = BernoulliRBM(random_state=0, verbose=True)
    #
    # classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    # # Hyper-parameters. These were set by cross-validation,
    # # using a GridSearchCV. Here we are not performing cross-validation to
    # # save time.
    # rbm.learning_rate = 0.06
    # rbm.n_iter = 20
    # # More components tend to give better prediction performance, but larger
    # # fitting time
    # rbm.n_components = 100
    # logistic.C = 6000.0
    #
    # # Training RBM-Logistic Pipeline
    # classifier.fit(X_train, Y_train)
    #
    # # Training Logistic regression
    # logistic_classifier = linear_model.LogisticRegression(C=100.0)
    # logistic_classifier.fit(X_train, Y_train)
    #
    # print()
    # print("Logistic regression using RBM features:\n%s\n" % (
    #     metrics.classification_report(
    #         Y_test,
    #         classifier.predict(X_test))))
    #
    # print("Logistic regression using raw pixel features:\n%s\n" % (
    #     metrics.classification_report(
    #         Y_test,
    #         logistic_classifier.predict(X_test))))

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

if __name__ == '__main__':
    np.set_printoptions(threshold=np.NaN)
    trainX = []
    trainy = []
    for i in range(4):
        f = 'data'+str(i)+'.npz'
        r = np.load(f)
        trainX.append(scale(r['X']))
        trainy.append(r['y'])
    trainX = np.concatenate((trainX[0], trainX[1], trainX[2], trainX[3]))
    trainY = np.concatenate((trainy[0], trainy[1], trainy[2], trainy[3]))
    r = np.load("test.npz")
    testX = scale(r['X'])
    testY = r['y']
    find_hyperparameter(trainX, trainY)
    # test2(trainX,trainY,testX,testy)
    # test_model(testX, testY, logistic_regression(trainX, trainY))
    # test_model(testX, testY, rbm_lr(trainX, trainY))
    # test_mnist()