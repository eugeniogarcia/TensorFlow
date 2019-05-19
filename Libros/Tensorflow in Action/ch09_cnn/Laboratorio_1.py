# -*- coding: utf-8 -*-
"""
Created on Sat May 18 07:56:40 2019

@author: Eugenio
"""

import numpy as np

a=np.array([1,2,3,3,2,1,3,2,1,1,2,3])

print(a.shape)

b=a
b=np.vstack((b,a))
b=np.vstack((b,a))
b=np.vstack((b,a))

print(b.shape)

c=b.reshape(b.shape[0],3,2,2)

print(c.shape)

d=c.mean(1)

print(d.shape)

e=d.reshape(d.shape[0],-1)

print(e.shape)

print(e)
print(e.mean(axis=0))
print(e.mean(axis=1))
