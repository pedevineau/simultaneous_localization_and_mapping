import numpy as np
import numpy.linalg as npl
import scipy.io
import math
import matplotlib.pyplot as plt
from slam import Robot

if __name__ == '__main__':
    '''
    Demo 1: show that odometry alone is strongly inaccurate
    '''
    series_nb = 8
    robot_nb = 1
    path = 'data/MRCLAM' + str(series_nb) + '.mat'
    data = scipy.io.loadmat(path)
    odom_str = 'Robot' + str(robot_nb) +'_Odometry'
    meas_str = 'Robot' + str(robot_nb) + '_Measurement'
    ground_str = 'Robot' + str(robot_nb) + '_Groundtruth'
    odom = data[odom_str][:, 1:3]
    meas = data[meas_str]
    ground = data[ground_str]
    barcodes = data['Barcodes'][:,1]
    landmarks = data['Landmark_Groundtruth'][:, 1:3]
    r = Robot(0, ground[0][1], ground[0][2], ground[0][3])
    r.plot_trajectory(odom, 0.02, landmarks, meas, ground, barcodes, landmarks, without_observation=True)
