#-*- coding:utf-8 -*-
import numpy as np
import preprocess

def scale(X, eps=0.001):
    # scale the data points s.t the columns of the feature space
    # (i.e the predictors) are within the range [0, 1]
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) + eps)


if __name__ == '__main__':
    a = 1+1