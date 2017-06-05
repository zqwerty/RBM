#-*- coding:utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM, MLPClassifier
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
    # find parameter
    # trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.1, random_state=42)
    # # perform a grid search on the 'C' parameter of Logistic
    # # Regression
    # print "SEARCHING LOGISTIC REGRESSION"
    # params = {"C": [1.0, 10.0, 100.0]}
    # start = time.time()
    # gs = GridSearchCV(LogisticRegression(), params, n_jobs=-1, verbose=1)
    # gs.fit(trainX, trainY)
    #
    # # print diagnostic information to the user and grab the
    # # best model
    # print "done in %0.3fs" % (time.time() - start)
    # print "best score: %0.3f" % (gs.best_score_)
    # print "LOGISTIC REGRESSION PARAMETERS"
    # bestParams = gs.best_estimator_.get_params()
    #
    # # loop over the parameters and print each of them out
    # # so they can be manually set
    # for p in sorted(params.keys()):
    #     print "\t %s: %f" % (p, bestParams[p])

    logistic = LogisticRegression(C=1.0)
    logistic.fit(trainX, trainY)
    print logistic.n_iter_
    return logistic


def rbm_lr(trainX, trainY):
    # # find parameter
    # # initialize the RBM + Logistic Regression pipeline
    # rbm = BernoulliRBM()
    # logistic = LogisticRegression()
    # classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
    #
    # # perform a grid search on the learning rate, number of
    # # iterations, and number of components on the RBM and
    # # C for Logistic Regression
    # print "SEARCHING RBM + LOGISTIC REGRESSION"
    # params = {
    #     "rbm__learning_rate": [0.001],
    #     "rbm__n_iter": [20, 80],
    #     "rbm__n_components": [100],
    #     "logistic__C": [100000, 1000000]}
    #
    # # perform a grid search over the parameter
    # start = time.time()
    # gs = GridSearchCV(classifier, params, n_jobs=-1, verbose=1)
    # gs.fit(trainX, trainY)
    #
    # # print diagnostic information to the user and grab the
    # # best model
    # print "\ndone in %0.3fs" % (time.time() - start)
    # print "best score: %0.3f" % (gs.best_score_)
    # print "RBM + LOGISTIC REGRESSION PARAMETERS"
    # bestParams = gs.best_estimator_.get_params()
    #
    # # loop over the parameters and print each of them out
    # # so they can be manually set
    # for p in sorted(params.keys()):
    #     print "\t %s: %f" % (p, bestParams[p])
    #
    rbm = BernoulliRBM(n_components=200, n_iter=20,
                       learning_rate=0.001, verbose=True)
    logistic = LogisticRegression(C=100000.0)

    # train the classifier and show an evaluation report
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
    classifier.fit(trainX, trainY)
    print logistic.n_iter_
    # print rbm.components_.shape
    # print rbm.fit_transform(trainX)[:10]
    return classifier


def rbm_svc(trainX, trainY):
    # # find parameter
    # # initialize the RBM + Logistic Regression pipeline
    # rbm = BernoulliRBM()
    # svc = SVC()
    # classifier = Pipeline([("rbm", rbm), ("svc", svc)])
    #
    # # perform a grid search on the learning rate, number of
    # # iterations, and number of components on the RBM and
    # # C for Logistic Regression
    # print "SEARCHING RBM + LOGISTIC REGRESSION"
    # params = {
    #     "rbm__learning_rate": [0.001],
    #     "rbm__n_iter": [20],
    #     "rbm__n_components": [200],
    #     "svc__C": [2000.0,5000,10000],
    #     "svc__kernel": ['linear', 'rbf', 'poly', 'sigmoid']}
    #
    # # perform a grid search over the parameter
    # start = time.time()
    # gs = GridSearchCV(classifier, params, n_jobs=-1, verbose=1)
    # gs.fit(trainX, trainY)
    #
    # # print diagnostic information to the user and grab the
    # # best model
    # print "\ndone in %0.3fs" % (time.time() - start)
    # print "best score: %0.3f" % (gs.best_score_)
    # print "RBM + LOGISTIC REGRESSION PARAMETERS"
    # bestParams = gs.best_estimator_.get_params()
    #
    # # loop over the parameters and print each of them out
    # # so they can be manually set
    # for p in sorted(params.keys()):
    #     print "\t %s: %s" % (p, str(bestParams[p]))

    rbm = BernoulliRBM(n_components=200, n_iter=20,
                       learning_rate=0.001, verbose=True)
    svc = SVC(C=2000, kernel='linear')
    classifier = Pipeline([("rbm", rbm), ("svc", svc)])
    classifier.fit(trainX, trainY)
    return classifier


def mlp(trainX, trainY):
    mlp = MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000, alpha=1e-4,
                        solver='sgd', verbose=True, tol=1e-4, random_state=1,
                        learning_rate_init=.001)
    mlp.fit(trainX, trainY)
    print("Training set score: %f" % mlp.score(trainX, trainY))
    print("Test set score: %f" % mlp.score(testX, testY))
    # print mlp.coefs_[0].shape
    # print mlp.intercepts_[0].shape
    return mlp


def rbm_mlp(trainX, trainY):
    rbm = BernoulliRBM(n_components=200, n_iter=20,
                       learning_rate=0.001, verbose=True)
    rbm.fit(trainX, trainY)
    W = np.transpose(rbm.components_)
    b = rbm.intercept_hidden_
    mlp = MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000, alpha=1e-4,
                        solver='sgd', verbose=True, tol=1e-4, random_state=1,
                        learning_rate_init=.001)
    mlp.fit(trainX[:1], trainY[:1])
    mlp.coefs_[0] = W #np.zeros((1024,200),dtype='float64')
    mlp.intercepts_[0] = b
    mlp.fit(trainX, trainY)
    return mlp


# class dbm

def dbm(trainX, trainY):
    rbm1 = BernoulliRBM(n_components=200, n_iter=20,
                        learning_rate=0.001, verbose=True)
    X_new = rbm1.fit_transform(trainX)
    print X_new[:10]
    # print X_new.shape
    rbm2 = BernoulliRBM(n_components=50, n_iter=200,
                        learning_rate=0.01, verbose=True)
    # pred_Y = rbm2.fit_transform(X_new)
    pred_Y = rbm2.fit_transform(X_new)
    # rbm2.fit(trainX)
    # W =
    print pred_Y[:10]
    rbm3 = BernoulliRBM(n_components=10, n_iter=200,
                        learning_rate=0.01, verbose=True)
    pred_Y = rbm2.fit_transform(X_new)
    print pred_Y[:10]
    # print pred_Y.shape, trainY.shape
    # for i in range(trainY.shape[0]):
    #     pred_y = list(pred_Y[i]).index(max(pred_Y[i]))
        # print pred_Y[i]
        # print pred_y, trainY[i]


    logistic = LogisticRegression(C=1.0)
    logistic.fit(pred_Y, trainY)
    print logistic.n_iter_
    return logistic


def dbm2(trainX, trainY, hidden_layer=(600, 200)):
    # # find parameter
    # # initialize the RBM + Logistic Regression pipeline
    # pipe = []
    # for i in range(len(hidden_layer)):
    #     rbm = BernoulliRBM(n_components=hidden_layer[i], n_iter=20,
    #                        learning_rate=0.001, verbose=True)
    #     pipe.append(("rbm" + str(i), rbm))
    # logistic = LogisticRegression()
    # # logistic = SVC(C=2000, kernel='linear')
    # pipe.append(("logistic", logistic))
    # classifier = Pipeline(pipe)
    #
    # # perform a grid search on the learning rate, number of
    # # iterations, and number of components on the RBM and
    # # C for Logistic Regression
    # print "SEARCHING RBM + LOGISTIC REGRESSION"
    # params = {
    #     "logistic__C": [100000000, 1000000000]}
    #
    # # perform a grid search over the parameter
    # start = time.time()
    # gs = GridSearchCV(classifier, params, n_jobs=-1, verbose=1)
    # gs.fit(trainX, trainY)
    #
    # # print diagnostic information to the user and grab the
    # # best model
    # print "\ndone in %0.3fs" % (time.time() - start)
    # print "best score: %0.3f" % (gs.best_score_)
    # print "RBM + LOGISTIC REGRESSION PARAMETERS"
    # bestParams = gs.best_estimator_.get_params()
    #
    # # loop over the parameters and print each of them out
    # # so they can be manually set
    # for p in sorted(params.keys()):
    #     print "\t %s: %f" % (p, bestParams[p])

    pipe = []
    for i in range(len(hidden_layer)):
        rbm = BernoulliRBM(n_components=hidden_layer[i], n_iter=20,
                       learning_rate=0.001, verbose=True)
        pipe.append(("rbm"+str(i), rbm))
    logistic = LogisticRegression(C=100000000.0)
    # logistic = SVC(C=2000, kernel='linear')
    pipe.append(("logistic", logistic))
    classifier = Pipeline(pipe)
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
    # X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    X = scale(X)
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


def load_all():
    trainX = []
    trainy = []
    for i in range(4):
        f = 'data' + str(i) + '.npz'
        r = np.load(f)
        trainX.append(scale(r['X']))
        trainy.append(r['y'])
    trainX = np.concatenate((trainX[0], trainX[1], trainX[2], trainX[3]))
    trainY = np.concatenate((trainy[0], trainy[1], trainy[2], trainy[3]))
    r = np.load("test.npz")
    testX = scale(r['X'])
    testY = r['y']
    return trainX, trainY, testX, testY

if __name__ == '__main__':
    np.set_printoptions(threshold=np.NaN)
    trainX, trainY, testX, testY = load_all()
    # find_hyperparameter(trainX, trainY)
    # test2(trainX,trainY,testX,testy)
    # test_model(testX, testY, logistic_regression(trainX, trainY))
    # test_model(testX, testY, rbm_lr(trainX, trainY))
    # test_model(testX, testY, rbm_svc(trainX, trainY))
    # test_model(testX, testY, mlp(trainX, trainY))
    # test_model(testX, testY, rbm_mlp(trainX, trainY))
    # dbm(trainX, trainY)
    test_model(testX, testY, dbm2(trainX, trainY))
    # test_mnist()