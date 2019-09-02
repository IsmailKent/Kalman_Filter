
import math
import numpy as np
import matplotlib.pyplot   as plt
import numpy.linalg as la

from Sensor import Sensor
from MovingObject import MovingObject


class Filter:
    sensor = None
    predictedCenters = None
    predictedPs = None
    filteredCenters = None
    filteredPs = None
    
    def __init__(self,sensor):
        self.sensor = sensor
        self.predictedCenters = [[]]
        self.predictedPs = [[]]
        self.filteredCenters = [[]]
        self.filteredPs = [[]]
    
    
    def Predict(self,oldCenter,oldP):
        F = self.sensor.createF()
        D = self.sensor.createD()
        centerPredicted = F @ oldCenter
        pPredicted = F @ oldP @ F.T + D
        self.predictedCenters.extend([centerPredicted])
        self.predictedPs.extend([pPredicted])
        return centerPredicted,pPredicted 

    def Filter(self,centerPredicted, pPredicted,currentTrajectory):
        R = self.sensor.createR()
        H = self.sensor.getH()
        S = H @ pPredicted @ H.T + R
        v = np.array([currentTrajectory]).T - H@ centerPredicted
        W = pPredicted @ H.T @ la.inv(S)
        newP = pPredicted - W@S@W.T
        newCenter = centerPredicted + W@v
        self.filteredCenters.extend([newCenter])
        self.filteredPs.extend([newP])
        return newCenter, newP
    
    def Retrodict(self,retrodictedX,retrodictedP,l): # X, l index, P l+1|k
        m , _ = self.sensor.createPo(1).shape
        l+=1
        Plgl = self.filteredPs[l] # p of l given l
        Plpogl = self.filteredPs[l+1] # p of l+1 given l
        Xlgl = self.filteredCenters[l]
        Xlpogl = self.filteredCenters[l+1]
        W = Plgl @ self.sensor.createF() @ Plpogl.T
        newRetrodictedP = Plgl + W @ (retrodictedP - Plpogl) @ W.T
        newRetrodictedCenter = Xlgl + W @ (retrodictedX - Xlpogl)
        return newRetrodictedCenter, newRetrodictedP
    
    
    
    