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
from prediction.msg import TrackedPoint, PredictedObjectPath, PredictedObjectPathList, TrackedObjectPose, TrackedObjectPoseList, PredictedHMM
import scipy
from scipy.stats import norm, multivariate_normal
from filter import Extended_KalmanFilter, IMM_filter
import numpy as np
from hmmlearn import hmm
import json
from std_msgs.msg import String
# from frame_transform import *

from model import CTRA, CA


##################          HMM MODEL           ###############
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
            [0.7, 0.15, 0.15], # Lane Keeping -> Others
            [0.6, 0.2,  0.2],  # Right Lane Change -> Others
            [0.6, 0.2,  0.2]   # Left Lane Changes -> Others
        ])

        # HMM 모델 생성
        self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full", init_params="")
        self.model.startprob_ = self.start_probabilities
        self.model.transmat_ = self.transition_matrix

        # Covariance
        self.model.covars_ = np.full((self.n_states, 1,1), 0.01)

        # MEANS
        self.model.means_ = np.zeros((self.n_states, 1))

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
        print("Emission matrix", emission_probabilities)

        return emission_probabilities

    def fit(self, observations):
        try:
            # 관측값 (d_lat, v) 시퀀스에서 방출 확률 계산
            emission_probs = np.array([self.calculate_emission_probabilities(d_lat, v) for d_lat, v in observations])
            if not emission_probs.size:
                raise ValueError("Empty emission probabilities")
            # HMM의 means와 covariances 설정 (방출 확률을 기준으로 함)
            self.model.means_ = np.mean(emission_probs, axis=0).reshape(-1, 1)
            #self.model.covars_ = np.var(emission_probs, axis=0).reshape(-1, 1, 1)
            self.model.covars_ = np.array([np.diag(np.var(emission_probs, axis=0) + 0.0001)])
            # 모델 적합
            self.model.fit(emission_probs.reshape(-1,1))

        except Exception as e:
            rospy.logerr(f"Failed to fit model: {e}")

    def predict(self, observations):
        try:
            if observations and all(isinstance(obs, tuple) and len(obs) ==2 for obs in observations):
                # 관측값 (d_lat, v) 시퀀스에서 방출 확률 계산
                emission_probs = np.array([self.calculate_emission_probabilities(d_lat, v) for d_lat, v in observations])
                if emission_probs.size == 0:
                    raise ValueError("No valid observation")
                # Viterbi 알고리즘을 이용하여 가장 가능성 높은 상태 시퀀스를 계산
                logprob, states = self.model.decode(emission_probs.reshape(-1,1), algorithm="viterbi")

                return logprob, states
                #return state_sequence
            else:
                raise ValueError("Invalid observation data")

        except Exception as e:
            rospy.logerr(f"Prediction failed: {e}")
            return None, []

################      Vehicle Tracker     ##############
class VehicleTracker():
    def __init__(self,dt=0.1, T=1):
        '''
            PredictedHMM.msg
            Header header
            int32 unique_id
            String maneuver
            float64 probability
        '''
        rospy.init_node('VehicleTracker', anonymous=True)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_info_callback)
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        self.state_prediction = rospy.Publisher("/Object_topic/hmm_prediction", PredictedHMM, queue_size= 10)

        self.dt=dt
        self.T=T
        self.v_max= 60.0 # Maximum Velocity :: Parameter Tunning
        self.rate=rospy.Rate(30)
        self.lane_width=3.521 # initial value

        ## Message Info
        self.is_object = False
        self.is_global_path = False
        self.object_data = None

        self.previous_heading = {}
        self.previous_time = rospy.Time.now()

        # Get Global Path
        self.global_path_msg = Path()
        self.global_path_msg.header.frame_id = 'map'
        self.x = 0
        self.y = 0
        self.yaw = 0
        # Managing Object
        self.hmm_models = {}


    ##########   CALLBACK Function   ########
    # Object Callback Function, Get the object info
    def object_info_callback(self, msg):
        #rospy.loginfo("Received Message")
        self.is_object=True
        self.object_data = msg

    def global_path_callback(self, msg):
        self.global_path_msg = msg
        self.is_global_path = True

    ##########   UTILITY     #######
    #TODO(2) : Data Preprocessing, Get information about the object data
    def data_preprocessing(self, obstacle, delta_time):
        obj_id = obstacle.unique_id

        # Calculating Velocity -> You have to Use Longitudinal Velocity
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
        print("Possibility of p_lk ", p_lkv)
        print("Possibility of left lane change" , p_lcv)
        print("Possibility of right lane change", p_lcv_minus)

        return p_lcv, p_lcv_minus, p_lkv, p_lk, p_lc

    ## MAIN PROGRAM ##
    def run(self):
        while not rospy.is_shutdown():
            predicted_list = []
            self.hmm_model = VehicleBehaviorHMM(self.lane_width, self.v_max)
            if self.is_object == True & self.is_global_path == True:
                print("Received Global Path for Prediction")
                current_time = rospy.Time.now()
                delta_time = (current_time - self.previous_time).to_sec()
                self.previous_time = current_time
                observations = {}

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

                    # Normalize d value (frenet coordinate value of each object(vehicle))
                    if d>0:
                        d_int = d // self.lane_width
                        d = d - d_int * self.lane_width

                    if d<0:
                        d_int = d// self.lane_width +1
                        d = d - d_int * self.lane_width

                    # Probability, b_ij, #Emission Possibility
                    # Observation Sequence (d_lat, v)

                    observation = (d, data[3])
                    print("observation:: ", observation)
                    #TODO(5) : HMM Model and Publish the TOPIC
                    if obj_id not in self.hmm_models:
                        self.hmm_models[obj_id] = VehicleBehaviorHMM(self.lane_width, self.v_max)
                    # Observations dictionary에 observation값 추가. time stamp관리?
                    # observations.append(observation)
                    #self.hmm_models[obj_id].fit([observations])
                    logprob, state_sequence = self.hmm_models[obj_id].predict([observation])

                    if state_sequence.size>0:
                        predicted_state = self.hmm_models[obj_id].states[state_sequence[0]]
                        probability = np.exp(logprob)
                        predicted_object = PredictedHMM(unique_id = obj_id, maneuver=predicted_state, probability=probability)
                        predicted_list.append(predicted_object)
                        # prediction_msg = f"Object ID: {obj_id}, Predicted State: {predicted_state}, Probability: {probability:.4f}"
                        print("prediction message", predicted_list)
                        # Publish ROS TOPIC
                        self.state_prediction.publish(predicted_object)
                    else:
                        rospy.logerr("ERROR:: Not Effective Data Type or State Sequence")
                self.rate.sleep()

    def callback_result(self, data):
        s_value, d_value= self.frenet_frame(data)
        for i in range(len(self.vehicles)):
            veh_data=np.array(self.vehicles[i][self.time-10:self.time+1])


###################         UTILS FUNCTION            ###############
def next_waypoint(x, y, mapx, mapy):
    closest_wp = 0
    closest_wp = get_closest_waypoints(x, y, mapx, mapy)

    print(len(mapx))
    print(closest_wp, "\n\n")

    # if closest_wp >= len(mapx) or closest_wp < 0:
    #     closest_wp = 0

    map_x = mapx[closest_wp]
    map_y = mapy[closest_wp]

    heading = np.arctan2((map_y - y), (map_x - x))
    angle = np.abs(np.arctan2(np.sin(heading), np.cos(heading)))

    if angle > np.pi / 4:
        next_wp = (closest_wp + 1) % len(mapx)
        dist_to_next_wp = get_dist(x, y, mapx[next_wp], mapy[next_wp])

        if dist_to_next_wp < 5:
            closest_wp = next_wp

    return closest_wp


def get_closest_waypoints(x, y, mapx, mapy):
    min_len = 1e10
    closest_wp = -1

    for i in range(len(mapx)):
        _mapx = mapx[i]
        _mapy = mapy[i]
        dist = get_dist(x, y, _mapx, _mapy)

        if dist < min_len:
            min_len = dist
            closest_wp = i

    if closest_wp == -1:
        rospy.loginfo("Invalid waypoint")

    return closest_wp

def get_dist(x, y, _x, _y):
    return np.sqrt((x - _x) ** 2 + (y - _y) ** 2)


def get_frenet(x, y, mapx, mapy):
    next_wp = next_waypoint(x, y, mapx, mapy)
    prev_wp = next_wp - 1 if next_wp > 0 else len(mapx) - 1

    n_x = mapx[next_wp] - mapx[prev_wp]
    n_y = mapy[next_wp] - mapy[prev_wp]
    x_x = x - mapx[prev_wp]
    x_y = y - mapy[prev_wp]

    if n_x == 0 and n_y == 0:
        proj_x = x_x
        proj_y = x_y
        proj_norm = 0
    else:
        proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)
        proj_x = proj_norm * n_x
        proj_y = proj_norm * n_y

    frenet_d = get_dist(x_x, x_y, proj_x, proj_y)

    ego_vec = [x - mapx[prev_wp], y - mapy[prev_wp], 0]
    map_vec = [n_x, n_y, 0]
    d_cross = np.cross(ego_vec, map_vec)

    if d_cross[-1] > 0:
        frenet_d = -frenet_d

    frenet_s = 0
    for i in range(prev_wp):
        frenet_s += get_dist(mapx[i], mapy[i], mapx[i + 1], mapy[i + 1])

    frenet_s += get_dist(mapx[prev_wp], mapy[prev_wp], mapx[prev_wp] + proj_x, mapy[prev_wp] + proj_y)

    return frenet_s, frenet_d

def get_cartesian(s, d, mapx, mapy, maps):
    prev_wp = 0

    while (prev_wp < len(maps) - 2) and (s > maps[prev_wp + 1]):
        prev_wp += 1

    next_wp = np.mod(prev_wp + 1, len(mapx))

    dx = (mapx[next_wp] - mapx[prev_wp])
    dy = (mapy[next_wp] - mapy[prev_wp])

    heading = np.arctan2(dy, dx)  # [rad]

    seg_s = s - maps[prev_wp]
    seg_x = mapx[prev_wp] + seg_s * np.cos(heading)
    seg_y = mapy[prev_wp] + seg_s * np.sin(heading)

    perp_heading = heading + np.pi / 2
    x = seg_x + d * np.cos(perp_heading)
    y = seg_y + d * np.sin(perp_heading)

    return x, y, heading

def next_waypoint(x, y, mapx, mapy):
    closest_wp = get_closest_waypoints(x, y, mapx, mapy)

    print(closest_wp, "\n\n")


    map_x = mapx[closest_wp]
    map_y = mapy[closest_wp]

    heading = np.arctan2((map_y - y), (map_x - x))
    angle = np.abs(np.arctan2(np.sin(heading), np.cos(heading)))

    if angle > np.pi / 4:
        next_wp = (closest_wp + 1) % len(mapx)
        dist_to_next_wp = get_dist(x, y, mapx[next_wp], mapy[next_wp])

        if dist_to_next_wp < 5:
            closest_wp = next_wp

    return closest_wp

def get_closest_waypoints(x, y, mapx, mapy):
    min_len = 1e10
    closest_wp = -1

    for i in range(len(mapx)):
        _mapx = mapx[i]
        _mapy = mapy[i]
        dist = get_dist(x, y, _mapx, _mapy)

        if dist < min_len:
            min_len = dist
            closest_wp = i

    if closest_wp == -1:
        rospy.loginfo("Invalid waypoint")

    return closest_wp

def get_dist(x, y, _x, _y):
    return np.sqrt((x - _x) ** 2 + (y - _y) ** 2)

if __name__ == '__main__':
    try:
        tracker = VehicleTracker()
        tracker.run()
    except rospy.ROSInitException:
        pass
