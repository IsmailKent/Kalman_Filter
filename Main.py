

""" Programm by: Ismail Wahdan - 3120711"""



import math
import numpy as np
import matplotlib.pyplot   as plt
import numpy.linalg as la
from Sensor import Sensor
from Filter import Filter
from MovingObject import MovingObject






movingObject = MovingObject()
sensor = Sensor(movingObject)
myFilter = Filter(sensor)

m , n = movingObject.getStateX().shape

"""plot ground truth"""
position = movingObject.r
plt.scatter(position[0], position[1],s=10,label="Ground truth")
plt.title("2D Plane", fontsize=19)
plt.xlabel("x", fontsize=10)
plt.ylabel("y", fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=9)
plt.legend(loc='lower right')
plt.grid()
plt.show()

"""Plot Sensor measures"""
Zc = sensor.ProduceNoisedMeasuresCoordinations()
plt.scatter(Zc[:,0], Zc[:,1],s=10,label="noised measures zk")
plt.xlabel("x", fontsize=10)
plt.ylabel("y", fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=9)
plt.legend(loc='lower right')


"""End TEST Part 1"""

"""Zpk is (Zrk,ZPhik)"""

"""construct rs"""

F = sensor.createF()


D = sensor.createD()


""" Produce all filters and predictions for visualization"""
X = np.zeros((6,1))
allX= np.zeros((m,6,1)) # save Xs
P = sensor.createPo(10)
Trajectory = sensor.ProduceTrajectoryUsingAAndR()
myFilter.filteredCenters.extend([X])
myFilter.filteredPs.extend([P])
for i in range(m):
    centerPredicted,pPredicted = myFilter.Predict(X,P)
    X,P = myFilter.Filter(centerPredicted,pPredicted,(Trajectory.T)[i])
    allX[i]=X
rx_f = list(allX[:,0,:1].T.flat)
ry_f = list(allX[:,1,:2].T.flat)
"""print filtered"""
plt.scatter(rx_f,ry_f, s=10, label = "after filtering") # print filtered predictions
plt.legend(loc='lower right')



# Retrodiction

Retrocidted_Xs = np.zeros(allX.shape)
Xcopy = X.copy()
for i in range(m):
    l = m-i-1
    X, P = myFilter.Retrodict(X,P,l)
    Retrocidted_Xs[i]=X
rx_r = list(Retrocidted_Xs[:,0,:1].T.flat)
ry_r = list(Retrocidted_Xs[:,1,:2].T.flat)
rx_r[-1]= Xcopy[0,:][0]
ry_r[-1]= Xcopy[1,:][0]
""" TO FIX: STORING X AND P"""
"""print retrodiction"""
plt.scatter(rx_r,ry_r,s=10,label="after retrodiction") # print Retrodiction
plt.legend(loc='lower right')

""" COMPARE ACCURACIES WITH EUCLIDEAN DISTANCE"""

plt.figure()
plt.title("Distance error", fontsize=19)
plt.xlabel('Time', color='#1C2833')
plt.ylabel('Distance', color='#1C2833')
plt.plot(movingObject.periodInterval, la.norm(position.T - Zc, axis=1), label='distance from truth of measures')
plt.plot(movingObject.periodInterval, la.norm(position.T - np.array([rx_f,ry_f]).T, axis=1),linewidth=3.0, label='mean distance from truth after filtering')
plt.plot(movingObject.periodInterval,la.norm(position.T - np.roll(np.flip(np.array([rx_r,ry_r]).T,axis = 0),-1,axis=0), axis=1), label=' mean distance from truth after retrodiction')
plt.legend(loc='lower right')



""" Ex4.4 """

"""FUSING 5 SENSORS"""


Sensors = []
for i in range(5):
    newSensor = Sensor(movingObject)
    newSensor.setRandomPosition()
    Sensors.append(newSensor)


"""create 5 diff measures from 5 sensors"""
Z_Sensors = np.zeros((5,2,m))
for i in range(5):
    Z_Sensors[i] = Sensors[i].ProduceNoisedMeasuresCoordinations().T


Sensors[0].sigmaC = 10
Sensors[1].sigmaC = 20
Sensors[2].sigmaC = 30
Sensors[3].sigmaC=70
Sensors[4].sigmaC=80

""" first algo slide 14
 work on H and product rule """

def fusSensors1(Sensors):
    # does not work
    H_allSensors = np.vstack(Sensors[0].getH(),Sensors[1].getH(),Sensors[2].getH(),Sensors[3].getH(),Sensors[4].getH())
    Z_allSensors = np.vstack(Sensors[0].ProduceNoisedMeasuresCoordinations(),Sensors[1].ProduceNoisedMeasuresCoordinations(),
                             Sensors[2].ProduceNoisedMeasuresCoordinations(),Sensors[3].ProduceNoisedMeasuresCoordinations(),
                             Sensors[4].ProduceNoisedMeasuresCoordinations())
    def createRallSensors():
        R = np.zeros(tuple([5*x for x in Sensors[0].createR().shape]))
        R[0:2,0:2] = Sensors[0].createR()
        R[2:4,2:4] = Sensors[1].createR()
        R[4:6,4:6] =Sensors[2].createR()
        R[6:8,6:8] = Sensors[3].createR()
        R[8:10,8:10] =Sensors[4].createR()
        return R
    
    R_allSensors = createRallSensors()
    newSensorWithRandH = Sensor(movingObject)
    newSensorWithRandH.R = R_allSensors
    newSensorWithRandH.H = H_allSensors
    newFilter = Filter(newSensorWithRandH)
    #print(Z_allSensors)
    # create new Sensor with new propertioes then new filter
    # you get the idea
    
    




"""2nd Fusion algo slide 17 
 work on one effictive measurement
create R matrices for 5 sensors """

def fusSensors2(Sensors):
    R_Sensors = np.array([Sensors[0].createR(),Sensors[1].createR(),Sensors[2].createR(),Sensors[3].createR(),Sensors[4].createR()])
    

    def createRfromSensors(R_Sensors):
        R2 = R_Sensors.copy()
        R2 = [la.inv(matrix) for matrix in R2]
        R2 = sum(R2)
        R2 = la.inv(R2)
        return R2
    
    
    R= createRfromSensors(R_Sensors)
    newSensorWithR = Sensor(movingObject)
    newSensorWithR.R = R
    newFilter = Filter(newSensorWithR)
    
    newTrajectory = np.zeros(Trajectory.T.shape)
    for i in range(m):
        for j in range(5):
            newTrajectory[i] += la.inv(R_Sensors[j]) @ Z_Sensors[j].T[i]
        newTrajectory[i] = R @ newTrajectory[i]
    newTrajectory = newTrajectory.T
    
    X_2ndStrategy = np.zeros((6,1))
    allX_2ndStrategy= np.zeros((m,6,1)) # save Xs
    P_2ndStrategy = sensor.createPo(10)
    for i in range(m):
        centerPredicted, pPredicted = newFilter.Predict(X_2ndStrategy,P_2ndStrategy)
        X_2ndStrategy,P_2ndStrategy = newFilter.Filter(centerPredicted,pPredicted,(newTrajectory.T)[i])
        allX_2ndStrategy[i]=X_2ndStrategy
    rx_2ndStrategy_f = list(allX[:,0,:1].T.flat)
    ry_2ndStrategy_f = list(allX[:,1,:2].T.flat)
    plt.scatter(rx_2ndStrategy_f,ry_2ndStrategy_f) # print filtered predictions

