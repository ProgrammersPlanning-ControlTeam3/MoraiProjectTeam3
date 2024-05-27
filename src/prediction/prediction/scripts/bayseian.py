#!/usr/bin/env python3
# coding: utf-8
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../msg'))
sys.path.insert(0, '/home/henricus/final_project/src')
import numpy as np
import time
import rospy
from geometry_msgs.msg import Point
from math import sqrt, radians, pow
from nav_msgs.msg import Odometry, Path
from morai_msgs.msg import EgoVehicleStatus, ObjectStatus, ObjectStatusList, EventInfo
from morai_msgs.srv import MoraiEventCmdSrv
from std_msgs.msg import Int32
from prediction.msg import TrackedPoint, PredictedObjectPath, PredictedObjectPathList, TrackedObjectPose, TrackedObjectPoseList
import scipy
from scipy.stats import norm, multivariate_normal
from filter import Extended_KalmanFilter, IMM_filter
import numpy as np
from hmmlearn import hmm

from model import CTRA, CA

class VehicleTracker():
    def __init__(self,dt=0.1, T=1):
        rospy.init_node('VehicleTracker', anonymous=True)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_info_callback)
        self.mu=np.array([0.9, 0.05, 0.05]) # Initial Transition Matrix
        self.dt=dt
        self.T=T
        self.v_max= 80.0 # v_max 정하셈... m/s
        # Parameter, 원래 딥러닝 모델로 학습시켜야 함... 개노가다 뛸 각오 하셈
        self.action_matrix=np.array([0.8, 0.1,  0.1], # 1번째 state LK
                                    [0.6, 0.01, 0.39], # 2번째 state Right change
                                    [0.6, 0.39, 0.01]) # 3번째 state Left Change
        self.lane_width=3.5 # initial value

    ##########   CALLBACK Function   ########
    # Object Callback Function, Get the object info
    def object_info_callback(self, msg):
        #rospy.loginfo("Received Message")
        self.is_object=True
        self.object_data = msg


    ########## Calculate b_ij probability using Gaussian Probability Distribution ########
    def calculate_probability(self, d_lat, v, lane_width):
        #TODO(1): Calculating Observation Probability
        ## Lane Change Profile Probability
        self.p_lcv = lambda d_lat, v : multivariate_normal.pdf(v, (-(2/self.lane_width)**2*self.v_max*(d_lat-self.lane_width/2)**2+self.v_max),0.4) if d_lat > 0 else multivariate_normal.pdf(v, (-(2/self.lane_width)**2*self.v_max*(d_lat+self.lane_width/2)**2+self.v_max), 0.4)
        self.p_lcv_minus = lambda d_lat, v : multivariate_normal.pdf(v, (2/self.lane_width)**2*self.v_max*(d_lat+self.lane_width/2)**2-self.v_max,0.4) if d_lat < 0 else  multivariate_normal.pdf(v, (2/self.lane_width)**2*self.v_max*(d_lat-self.lane_width/2)**2-self.v_max, 0.4)

        ## Lane Keeping Profile Probability
        self.p_lkv = lambda x, v: multivariate_normal.pdf(v, (-(2/self.lane_width)**2*self.v_max*(x)**2),0.4) if x > 0 else multivariate_normal.pdf(v, ((2/self.w)**2*self.v_max*(x)**2),0.4)

        mu_lc=-(2/lane_width)**2 * self.v_max*(d_lat-lane_width/2)**2+self.v_max
        mu_lk=-(2/lane_width)**2 * self.v_max*d_lat**2

        # Variance
        sigma=1.1
        #P(A|C, M)
        # Normal Gaussian Probability Density Function
        p_lc_action = norm.pdf(v, mu_lc, sigma) # 속도에 대한 확률
        p_lk_action = norm.pdf(v, mu_lk, sigma) # 속도에 대한 확률

        #P(M|C)
        p_lk=norm.pdf(d_lat, 0, sigma) # lk 확률
        p_lc=norm.pdf(d_lat, lane_width/2,sigma) # lc 확률

        return p_lc_action, p_lk_action, p_lk, p_lc

    def frenet_frame(self, data):
        #TODO(2) : Get frenet frame coordinate for every vehicles.
        d_lat={}
        s_long={}
        return s_long, d_lat

    def callback_result(self, data):
        s_value, d_value= self.frenet_frame(data)
        for i in range(len(self.vehicles)):
            veh_data=np.array(self.vehicles[i][self.time-10:self.time+1])

        #TODO(3) : Prediction using Veh History Data (Markov Decision Problem)
        # P(A|C,M)P(M|C)/SUM(P(A|C,M)) = P(M|A,C) # Select Maneuver
        pred="LC" if np.argmax(pred_lk, pred_lc, pred_rc)==1 else "LK"

