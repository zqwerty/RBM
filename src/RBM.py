#-*- coding:utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
import time
import gzip
import cPickle


def scale(X, eps=0.001):
    # scale the data points s.t the columns of the feature space
    # (i.e the predictors) are within the range [0, 1]
    # return (X - np.min(X, axis=0)) / (np.max(X, axis=0) + eps)
    # return (X - np.min(X, axis=1)) / (np.max(X, axis=1) + eps)
    return X/255


def logistic_regression(trainX, trainY):
    logistic = LogisticRegression(C=1.0)
    logistic.fit(trainX, trainY)
    print logistic.n_iter_
    return logistic


def svc_model(trainX, trainY):
    svc = SVC(kernel='linear')
    svc.fit(trainX, trainY)
    return svc


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


def rbm_lr(trainX, trainY):
    rbm = BernoulliRBM(n_components=200, n_iter=20,
                       learning_rate=0.001, verbose=True)
    logistic = LogisticRegression(C=100000.0)

    # train the classifier and show an evaluation report
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
    classifier.fit(trainX, trainY)
    print logistic.n_iter_
    return classifier


def rbm_svc(trainX, trainY):
    rbm = BernoulliRBM(n_components=200, n_iter=20,
                       learning_rate=0.001, verbose=True)
    svc = SVC(C=2000, kernel='linear')
    classifier = Pipeline([("rbm", rbm), ("svc", svc)])
    classifier.fit(trainX, trainY)
    return classifier


def rbm_mlp(trainX, trainY):
    rbm = BernoulliRBM(n_components=200, n_iter=20,
                       learning_rate=0.001, verbose=True)
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10000, alpha=1e-4,
                        solver='sgd', verbose=True, tol=1e-4, random_state=1,
                        learning_rate_init=.001)
    classifier = Pipeline([("rbm", rbm), ("mlp", mlp)])
    classifier.fit(trainX, trainY)
    return classifier


def pca_lr(trainX, trainY):
    pca = IncrementalPCA(n_components=200)
    logistic = LogisticRegression(C=1.0)

    # train the classifier and show an evaluation report
    classifier = Pipeline([("PCA", pca), ("logistic", logistic)])
    classifier.fit(trainX, trainY)
    print logistic.n_iter_
    return classifier


def pca_svc(trainX, trainY):
    pca = IncrementalPCA(n_components=200)
    svc = SVC(kernel='linear')
    classifier = Pipeline([("PCA", pca), ("svc", svc)])
    classifier.fit(trainX, trainY)
    return classifier


def pca_mlp(trainX, trainY):
    pca = IncrementalPCA(n_components=200)
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10000, alpha=1e-4,
                        solver='sgd', verbose=True, tol=1e-4, random_state=1,
                        learning_rate_init=.001)
    classifier = Pipeline([("PCA", pca), ("mlp", mlp)])
    classifier.fit(trainX, trainY)
    return classifier


def rbm2(trainX, trainY, hidden_layer=(600, 200)):
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


def pca2(trainX, trainY, hidden_layer=(600, 200)):
    pipe = []
    for i in range(len(hidden_layer)):
        pca = IncrementalPCA(n_components=hidden_layer[i])
        pipe.append(("PCA"+str(i), pca))
    logistic = LogisticRegression(C=1.0)
    # logistic = SVC(C=2000, kernel='linear')
    pipe.append(("logistic", logistic))
    classifier = Pipeline(pipe)
    classifier.fit(trainX, trainY)
    return classifier


def load_mnist_test():
    '''
    载入 mnist 数据集
    :return:
    '''
    f = gzip.open('../dataset/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    trainX = train_set[0]
    trainY = train_set[1]
    testX = test_set[0]
    testY = test_set[1]
    f.close()
    # return trainX[:10000], trainY[:10000], testX[:1000], testY[:1000]
    return trainX, trainY, testX, testY


def load_all():
    '''
    载入预处理过的所给数据集
    :return:
    '''
    trainX = []
    trainy = []
    for i in range(4):
        f = '../dataset/data' + str(i) + '.npz'
        r = np.load(f)
        trainX.append(scale(r['X']))
        trainy.append(r['y'])
    trainX = np.concatenate((trainX[0], trainX[1], trainX[2], trainX[3]))
    trainY = np.concatenate((trainy[0], trainy[1], trainy[2], trainy[3]))
    r = np.load("../dataset/test.npz")
    testX = scale(r['X'])
    testY = r['y']
    return trainX, trainY, testX, testY


def test_model(testX, testY, model):
    return classification_report(testY, model.predict(testX))


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


if __name__ == '__main__':
    # np.set_printoptions(threshold=np.NaN)

    trainX, trainY, testX, testY = load_all()

    # find_hyperparameter(trainX, trainY)
    f = open('../report/result.txt', 'w')

    f.write('On given data set: train:720; test:100\n\n')

    f.write('\nLogistic Regression:\n')
    f.write(test_model(testX, testY, logistic_regression(trainX, trainY)))
    f.write('\nSVC:\n')
    f.write(test_model(testX, testY, svc_model(trainX, trainY)))
    f.write('\nMLP:\n')
    f.write(test_model(testX, testY, mlp(trainX, trainY)))

    f.write('\nRBM+LR:\n')
    f.write(test_model(testX, testY, rbm_lr(trainX, trainY)))
    f.write('\nRBM+SVC:\n')
    f.write(test_model(testX, testY, rbm_svc(trainX, trainY)))
    f.write('\nRBM+MLP:\n')
    f.write(test_model(testX, testY, rbm_mlp(trainX, trainY)))
    f.write('\nRBM+RBM+LR:\n')
    f.write(test_model(testX, testY, rbm2(trainX, trainY)))

    f.write('\nPCA+LR:\n')
    f.write(test_model(testX, testY, pca_lr(trainX, trainY)))
    f.write('\nPCA+SVC:\n')
    f.write(test_model(testX, testY, pca_svc(trainX, trainY)))
    f.write('\nPCA+MLP:\n')
    f.write(test_model(testX, testY, pca_mlp(trainX, trainY)))
    f.write('\nPCA+PCA+LR:\n')
    f.write(test_model(testX, testY, pca2(trainX, trainY)))

    f.write('\n\nOn MNIST dataset: train:50000; test:10000\n')

    trainX, trainY, testX, testY = load_mnist_test()

    f.write('\nLogistic Regression:\n')
    f.write(test_model(testX, testY, logistic_regression(trainX, trainY)))

    f.write('\nRBM+LR:\n')
    f.write(test_model(testX, testY, rbm_lr(trainX, trainY)))

    f.write('\nPCA+LR:\n')
    f.write(test_model(testX, testY, pca_lr(trainX, trainY)))

    f.close()
