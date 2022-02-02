# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 02:35:10 2022

@author: jkcle
"""
import numpy as np
import seaborn as sns
import scipy.stats as stats

mvn = stats.multivariate_normal([0, 0], [[1, 0.8],
                                      [0.8, 1]])

n = 1000

x = mvn.rvs(n)

h = sns.jointplot(x[:, 0], x[:, 1], kind='kde', stat_func=None);
h.set_axis_labels('X1', 'X2', fontsize=16);

norm = stats.norm()
x_unif = norm.cdf(x)
h = sns.jointplot(x_unif[:, 0], x_unif[:, 1], kind='hex')
h.set_axis_labels('Y1', 'Y2', fontsize=16);

m1 = stats.lognorm(s=0.05, scale=np.exp(0.03))
m2 = stats.lognorm(s=0.1, scale=np.exp(0.05))

x1_trans = m1.ppf(x_unif[:, 0])
x2_trans = m2.ppf(x_unif[:, 1])

h = sns.jointplot(x1_trans, x2_trans, kind='kde');
h.set_axis_labels('X', 'Y', fontsize=16);