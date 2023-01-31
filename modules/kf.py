import cv2
import numpy as np

'''KALMAN FILTERING CLASS FOR N 2D POINTS'''
'''Kalman filtering for selected points in an image using OpenCV cv2.kalmanFilter class in Python '''
class Kalman_Filtering:

    def __init__(self,n_points):
        self.n_points = n_points

    def initialize(self):

        n_states = self.n_points * 4
        n_measures = self.n_points * 2
        self.kalman = cv2.KalmanFilter(n_states,n_measures)
        kalman = self.kalman
        kalman.transitionMatrix = np.eye(n_states, dtype = np.float32)
        #kalman.processNoiseCov = np.eye(n_states, dtype = np.float32)*0.9
        kalman.measurementNoiseCov = np.eye(n_measures, dtype = np.float32)*0.0005

        kalman.measurementMatrix = np.zeros((n_measures,n_states), np.float32)
        dt = 1

        self.Measurement_array = []
        self.dt_array = []

        for i in range(0,n_states,4):
            self.Measurement_array.append(i)
            self.Measurement_array.append(i+1)

        for i in range(0,n_states):
            if i not in self.Measurement_array:
                self.dt_array.append(i)

        print(self.dt_array)
        print(self.Measurement_array)
        #Transition Matrix for [x,y,x',y'] for n such points
        # format of first row [1 0 dt 0 .....]
        for i, j in zip(self.Measurement_array, self.dt_array):
            kalman.transitionMatrix[i,j] = dt;

        #Measurement Matrix for [x,y,x',y'] for n such points
        # format of first row [1 0 0 0 .....]
        for i in range(0,n_measures):
            kalman.measurementMatrix[i,self.Measurement_array[i]] = 1


        print('TRANSITION Matrix:')
        print(kalman.transitionMatrix)

        print('MEASUREMENT Matrix:')
        print(kalman.measurementMatrix)



    def predict(self,points):

        pred = []
        input_points = np.float32(np.ndarray.flatten(points))
        #Correction Step
        self.kalman.correct(input_points)
        #Prediction step
        tp = self.kalman.predict()

        for i in self.Measurement_array:
            pred.append(int(tp[i]))

        return pred
'''
USAGE: points must be a 2d numpy array of points, e.g.
input points are:
[[ x1.  y1.]
 [ x2.  y2.]
 [ x3.  y3.]
 [ x4.  y4.]
 [ x5.  y5.]
 [ x6.  y6.]]


import kalman_class

kf = kalman_class.Kalman_Filtering(6)
kf.initialize()
...
...
kf.predict(points)


'''