
import math
import numpy as np
import matplotlib.pyplot   as plt
import numpy.linalg as la


class MovingObject:
    
    deltaT = 5
    v = 300
    q = 9
    A = v*v/q
    w = q/(2*v)
    period = 4*math.pi*v/q
    periodInterval = np.arange(0,period,deltaT)
    r = A * np.array([np.sin(w*periodInterval),np.sin(2*w*periodInterval)])
    vel = v * np.array([0.5*np.cos(w*periodInterval), np.cos(2*w*periodInterval)])
    acc = -q * np.array([0.25*np.sin(w*periodInterval),np.sin(2*w*periodInterval)])
    
    def getStateX(self):
        X = np.hstack((np.hstack((self.r.transpose(),self.vel.transpose())),self.acc.transpose()))
        return X
