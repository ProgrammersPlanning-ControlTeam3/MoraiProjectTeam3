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
import json
from frame_transform import *


from model import CTRA, CA

class VehicleBehaviorHMM:
    def __init__(self, lane_width, v_max):
        self.lane_width = lane_width
        self.v_max = v_max

        # 상태 정의
        self.states = ["Lane Keeping", "Right Lane Change", "Left Lane Change"]
        self.n_states = len(self.states)

        # 초기 상태 확률
        self.start_probabilities = np.array([0.6, 0.2, 0.2])

        # 상태 전이 확률 행렬
        self.transition_matrix = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6]
        ])

        # HMM 모델 생성
        self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full", init_params="")
        self.model.startprob_ = self.start_probabilities
        self.model.transmat_ = self.transition_matrix

    def calculate_emission_probabilities(self, d_lat, v):
        # Lane Change Profile Probability
        p_lcv = lambda d_lat, v : multivariate_normal.pdf(v, (-(2/self.lane_width)**2*self.v_max*(d_lat-self.lane_width/2)**2+self.v_max),0.4) if d_lat > 0 else multivariate_normal.pdf(v, (-(2/self.lane_width)**2*self.v_max*(d_lat+self.lane_width/2)**2+self.v_max), 0.4)
        p_lcv_minus = lambda d_lat, v : multivariate_normal.pdf(v, (2/self.lane_width)**2*self.v_max*(d_lat+self.lane_width/2)**2-self.v_max,0.4) if d_lat < 0 else  multivariate_normal.pdf(v, (2/self.lane_width)**2*self.v_max*(d_lat-self.lane_width/2)**2-self.v_max, 0.4)

        # Lane Keeping Profile Probability
        p_lkv = lambda x, v: multivariate_normal.pdf(v, (-(2/self.lane_width)**2*self.v_max*(x)**2),0.4) if x > 0 else multivariate_normal.pdf(v, ((2/self.lane_width)**2*self.v_max*(x)**2),0.4)

        # 방출 확률 계산
        emission_probabilities = np.array([
            p_lkv(d_lat, v),  # Lane Keeping
            p_lcv(d_lat, v),  # Right Lane Change
            p_lcv_minus(d_lat, v)  # Left Lane Change
        ])

        return emission_probabilities

    def fit(self, observations):
        # 관측값 (d_lat, v) 시퀀스에서 방출 확률 계산
        emission_probs = np.array([self.calculate_emission_probabilities(d_lat, v) for d_lat, v in observations])

        # HMM의 means와 covariances 설정 (방출 확률을 기준으로 함)
        self.model.means_ = np.mean(emission_probs, axis=0).reshape(-1, 1)
        self.model.covars_ = np.var(emission_probs, axis=0).reshape(-1, 1, 1)

        # 모델 적합
        self.model.fit(emission_probs)

    def predict(self, observations):
        # 관측값 (d_lat, v) 시퀀스에서 방출 확률 계산
        emission_probs = np.array([self.calculate_emission_probabilities(d_lat, v) for d_lat, v in observations])

        # Viterbi 알고리즘을 이용하여 가장 가능성 높은 상태 시퀀스를 계산
        logprob, state_sequence = self.model.decode(emission_probs, algorithm="viterbi")

        return state_sequence

class VehicleTracker():
    def __init__(self,dt=0.1, T=1):
        rospy.init_node('VehicleTracker', anonymous=True)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_info_callback)
        rospy.Subscriber("/global_path", Path, self.global_path_callback)

        self.dt=dt
        self.T=T
        self.v_max= 80.0 # v_max 정하셈... m/s
        self.rate=rospy.Rate(30)
        self.lane_width=3.521 # initial value
        self.is_object = False
        self.object_data = None
        self.previous_heading = {}
        self.previous_time = rospy.Time.now()

        self.global_path_msg = Path()
        self.global_path_msg.header.frame_id = 'map'
        self.x = 0
        self.y = 0
        self.yaw = 0



        # # Parameter, 인공지능 딥러닝 모델 만들어서 학습시켜야 함... 노가다 뛸 각오 하셈
        # self.states = ["Lane Keeping", "Right Lane Change", "Left Lane Change"]
        # self.mu=np.array([0.9, 0.05, 0.05]) # Initial Transition Matrix
        # self.action_transition_matrix=np.array([
        #                             [0.8, 0.1,  0.1], # 1번째 state LK
        #                             [0.6, 0.01, 0.39], # 2번째 state Right change
        #                             [0.6, 0.39, 0.01] # 3번째 state Left Change
        #                             ])
   
        # # self.velocity_observations = ["low speed", "high speed"]
        # # self.distance_observations = []
        # self.model = hmm.GaussianHMM(n_components=3, covariance_type= " full")
        # self.model.startprob_=self.mu
        # self.model.transmat_ = self.action_transition_matrix

    ##########   CALLBACK Function   ########
    # Object Callback Function, Get the object info
    def object_info_callback(self, msg):
        #rospy.loginfo("Received Message")
        self.is_object=True
        self.object_data = msg

    def global_path_callback(self, msg):
        self.global_path_msg = msg

    ##########   UTILITY     #######
    #TODO(2) : Data Preprocessing, Get information about the object data
    def data_preprocessing(self, obstacle, delta_time):
        obj_id = obstacle.unique_id

        # Calculating Velocity
        v = sqrt(pow(obstacle.velocity.x, 2) + pow(obstacle.velocity.y, 2))

        # Calculating ACC
        a = sqrt(pow(obstacle.acceleration.x, 2) + pow(obstacle.acceleration.y, 2))

        # 이전 heading과 비교하여 yaw rate 계산
        if obj_id in self.previous_heading:
            previous_heading = self.previous_heading[obj_id]
            yaw_rate = (radians(obstacle.heading) - previous_heading) / delta_time
        else:
            yaw_rate = 0

        data = [obstacle.position.x, obstacle.position.y, radians(obstacle.heading), v, a, yaw_rate]

        return data


    ########## Calculate b_ij probability using Gaussian Probability Distribution ########
    def calculate_probability(self, d_lat, v, lane_width):
        #b_ij : Certain Maneuver -> Observation Value, Not a state... This is continuous value
        # For given d_lat, velocity -> Maneuver (필요 충분 조건이라고 가정)
        # 입력값의 v와 d_lat은 각 object 차량의 d값과 속도값을 받아와야 함.
        #TODO(3): Calculating Observation Probability
        # P(A|C, M)
        ## Lane Change Profile Probability
        p_lcv = lambda d_lat, v : multivariate_normal.pdf(v, (-(2/self.lane_width)**2*self.v_max*(d_lat-self.lane_width/2)**2+self.v_max),0.4) if d_lat > 0 else multivariate_normal.pdf(v, (-(2/self.lane_width)**2*self.v_max*(d_lat+self.lane_width/2)**2+self.v_max), 0.4)
        p_lcv_minus = lambda d_lat, v : multivariate_normal.pdf(v, (2/self.lane_width)**2*self.v_max*(d_lat+self.lane_width/2)**2-self.v_max,0.4) if d_lat < 0 else  multivariate_normal.pdf(v, (2/self.lane_width)**2*self.v_max*(d_lat-self.lane_width/2)**2-self.v_max, 0.4)

        ## Lane Keeping Profile Probability
        p_lkv = lambda x, v: multivariate_normal.pdf(v, (-(2/self.lane_width)**2*self.v_max*(x)**2),0.4) if x > 0 else multivariate_normal.pdf(v, ((2/self.w)**2*self.v_max*(x)**2),0.4)

        #P(M|C)
        p_lk=norm.pdf(d_lat, 0, 0.4) # lk 확률
        p_lc=norm.pdf(d_lat, lane_width/2, 0.4) # lc 확률

        return p_lcv, p_lcv_minus, p_lkv, p_lk, p_lc

    ## MAIN PROGRAM ##
    def run(self):
        while not rospy.is_shutdown():
            self.hmm_model = VehicleBehaviorHMM(self.lane_width, self.v_max)
            if self.is_object == True:
                file_path = '/home/henricus/final_project/src/prediction/src/added_sensor_network.json'
                current_time = rospy.Time.now()
                delta_time = (current_time - self.previous_time).to_sec()
                #TODO(4) : Get Frenet Frame coordinate info of the other vehicles based on Ego Vehicle
                for obstacle in self.object_data.npc_list: # Get by callback function
                    obj_id = obstacle.unique_id
                    data = self.data_preprocessing(obstacle, delta_time) # Get from MORAI
                    # data = [obstacle.position.x, obstacle.position.y, radians(obstacle.heading), v, a, yaw_rate]
                    x = obstacle.position.x
                    y = obstacle.position.y
                    mapx = [pose.pose.position.x for pose in self.global_path_msg.poses]
                    mapy = [pose.pose.position.y for pose in self.global_path_msg.poses]
                    s,d = get_frenet(x, y, mapx, mapy)

                    # Normalize d value
                    if d>0:
                        d_int = d // self.lane_width
                        d = d - d_int * self.lane_width
                    if d<0:
                        d_int = d// self.lane_width +1
                        d = d - d_int * self.lane_width
                    # Probability, b_ij, #Emission Possibility
                    # Observation Sequence (d_lat, v)
                    observation = np.array([d, data[4]])
                    self.hmm_model.fit(observation)
                    state_sequence = self.hmm_model.predict(observation)
                    predicted_states = [self.hmm_model.states[state] for state in state_sequence]
                    print("Predicted States:")
                    print(predicted_states)


    def callback_result(self, data):
        s_value, d_value= self.frenet_frame(data)
        for i in range(len(self.vehicles)):
            veh_data=np.array(self.vehicles[i][self.time-10:self.time+1])

if __name__ == '__main__':
    try:
        tracker = VehicleTracker()
        tracker.run()
    except rospy.ROSInitException:
        pass
