# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 01:08:59 2022

@author: jkcle

Uses code from http://davmre.github.io/blog/python/2013/12/15/orthogonal_poly
"""
import numpy as np


def ortho_poly_fit(x, degree=1):
    n = degree + 1
    x = np.asarray(x).flatten()
    if(degree >= len(np.unique(x))):
        print(len(np.unique(x)))
        raise ValueError("'degree' must be less than number of unique points")
    xbar = np.mean(x)
    x = x - xbar
    X = np.fliplr(np.vander(x, n))
    q,r = np.linalg.qr(X)

    z = np.diag(np.diag(r))
    raw = np.dot(q, z)

    norm2 = np.sum(raw**2, axis=0)
    alpha = (np.sum((raw**2)*np.reshape(x,(-1,1)), axis=0)/norm2 + xbar)[:degree]
    Z = raw / np.sqrt(norm2)

    return Z, norm2, alpha


def ortho_poly_predict(x, norm2, alpha, degree = 1):
    x = np.asarray(x).flatten()
    n = degree + 1
    Z = np.empty((len(x), n))
    Z[:,0] = 1
    if degree > 0:
        Z[:, 1] = x - alpha[0]
    if degree > 1:
        for i in np.arange(1,degree):
            Z[:, i+1] = (x - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i-1]) * Z[:, i-1]
    Z /= np.sqrt(norm2)
    return Z

class OrthoPolyReg():
    def __init__(self):
        return
    def fit(self, x, y, degree):
        self.degree = degree
        X, self.norm2, self.alpha = ortho_poly_fit(x, degree=degree)
        self.beta = np.linalg.lstsq(X, y, rcond=None)[0]
        self.resids = y - (X @ self.beta).flatten()
    def predict(self, x):
        X = ortho_poly_predict(x, self.norm2, self.alpha, degree=self.degree)
        return X @ self.beta