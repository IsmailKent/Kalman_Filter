
import math
import numpy as np
import matplotlib.pyplot   as plt
import numpy.linalg as la

from MovingObject import MovingObject


class Sensor:
    rs = np.zeros((2,1))
    sigmaC = 50
    deltaT = 5
    sigmaC = 50
    sigmaR = 20
    sigmaPhi = np.deg2rad(0.2) # degree, take care of rad calculations
    myObject = None
    X = None
    H = None
    R = None
    m =0
   
    
    
    def __init__(self,myObject):
        self.myObject = myObject
        self.X = myObject.getStateX()
        self.m , _ = self.X.shape
    
    def createF(self):
        F = np.eye(6)
        F[0:2,2:4] = self.deltaT * np.eye(2)
        F[0:2,4:6] = 0.5*self.deltaT**2 * np.eye(2)
        F[2:4,4:6] = self.deltaT * np.eye(2)
        return F


    def createD(self):
        F = self.createF()
        G = F[0:6,4:6]
        return self.sigmaC **2 * G@G.T

    def createPo(self,diag):
        return diag * np.eye(6)
    
    def createR(self):
        if not (self.R is None):
            return self.R
        self.R = self.sigmaC * np.eye(2)
        return self.R
    
    def getH(self):
        if not (self.H is None):
            return self.H
        H = np.hstack((np.hstack((np.eye(2),np.zeros((2,2)))),np.zeros((2,2))))
        return H
    
    
    def ProduceNoisedMeasuresCoordinations(self):
        X = self.myObject.getStateX()
        Zc = np.zeros((self.m,2))
        H = self.getH()
        for i in range(self.m):
            Zc[i] = H@X[i] + self.sigmaC * np.random.normal(0, 1, 2)
        return Zc
    
    def ProduceNoisedMeasuresAngleAndR(self):
        r = self.myObject.r
        Zr = np.zeros(self.m)
        ZPhi = np.zeros(self.m)
        for i in range(self.m):
            xk = r[0][i]
            xs = self.rs[0][0]
            yk = r[1][i]
            ys=self.rs[1][0]
            Zr[i] = np.sqrt((xk-xs)**2 + (yk-ys)**2) + self.sigmaR * np.random.normal(0, 1)
            if (xk-xs!=0):
                ZPhi[i] = np.arctan2(yk-ys,xk-xs) + self.sigmaPhi * np.random.normal(0, 1)
            else:
                ZPhi[i]= 0
        return np.array(([Zr,ZPhi]))
    
    def ProduceTrajectoryUsingAAndR(self):
        Zr , ZPhi = self.ProduceNoisedMeasuresAngleAndR()
        cosZPhi = np.cos(ZPhi)
        sinZPhi = np.sin(ZPhi)
        Trajectory = np.array([Zr*cosZPhi,Zr*sinZPhi]) + self.rs
        return Trajectory
    
    def setRandomPosition(self):
        self.rs = np.random.random_integers(1,2000, size=(2,1))