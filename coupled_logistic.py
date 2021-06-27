# -*- coding: utf-8 -*-
"""
"""
from __future__ import division
from numpy import zeros
import numpy as np
from pylab import*
import pylab as pl

N = 200 # the number of maps
a = 4.0 # the nonlinear parameter of the logistic map
eps = 0.4995 # the coupling strength
   
fun = lambda u: a*u*(1-u) #the logistic map
    
def f1(x):
    v = fun(x)
    dx = (1 - eps)*v + eps*np.mean(v)
    return dx


N1 = 500010
x0=np.loadtxt('initial.dat',float)
#x0 = np.random.uniform(0,1.0,N)
sol = np.zeros([N1+1,N],float)
sol[0,:] = x0

for k2 in range (0,N1):
    print(k2)
    u = f1(sol[k2,:])
    sol[k2+1,:] = u
