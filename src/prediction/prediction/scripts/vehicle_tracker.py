#!/usr/bin/env python3
# coding: utf-8
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../msg'))
sys.path.insert(0, '/home/ubuntu/MoraiProjectTeam3/src')
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

# This is the physics based Model.
# But we only need to decide the... Decision Based Model
class DynamicObstacleTracker:
  def __init__(self, dt=0.1, T=1):
        mat_trans = np.array([[0.85, 0.15],
                              [0.15, 0.85]])

        mu = [0.8, 0.2]

        self.dt = dt
        self.T = T

        self.filters = [Extended_KalmanFilter(5, 4),
                        Extended_KalmanFilter(6, 4)]

        self.models = [CA(self.dt), CTRA(self.dt)]

        self.Q_list = [[0.5, 0.5, 0.5, 0.5, 0.05],
                       [0.5, 0.5, 0.5, 0.5, 0.05, 0.01]] # process noise, Should fix, I just copied

        for i in range(len(self.filters)):
            self.filters[i].F = self.models[i].step
            self.filters[i].H = self.models[i].H
            self.filters[i].JA = self.models[i].JA
            self.filters[i].JH = self.models[i].JH
            self.filters[i].Q = np.diag(self.Q_list[i])
            self.filters[i].R = np.diag([0.05, 0.05, 0.05, 0.05]) # measurement noise
        # IMM filter 예측
        self.IMM = IMM_filter(self.filters, mu, mat_trans)
        self.MM = [mu]
        self.X = [] # Solution

  def initialize(self, data):
    x=[np.array([data[0], data[1], data[3], 0, data[2]]),
                np.array([data[0], data[1], data[3], 0, data[2], 0])]
    for i in range(len(self.filters)):
      self.filters[i].x=x[i]

    self.X.append(x[1])

  def update(self, data):
    z=[data[0], data[1], data[3], data[2]]
    self.IMM.prediction()
    self.IMM.merging(z)

    while len(self.MM) > int(self.T/self.dt):
      self.MM.pop(0)
    while len(self.X) > int(self.T/self.dt):
      self.X.pop(0)

    self.MM.append(self.IMM.mu.copy())
    self.X.append(self.IMM.x.copy())

    # 예측 코드 구현
  def predict(self):
    traj = self.IMM.predict(self.T)
    return traj


class MultiDynamicObstacleTracker:
    def __init__(self, dt=0.1, T=1, timeout=1.5):
        self.trackers = {} # 트랙커 배열 정의
        self.dt = dt # 관측 시간 정의
        self.T = T # 토털 시간 정의
        self.timeout = timeout  # Object Timeout (seconds)
        self.deleted_ids=set()

    def add_tracker(self, obj_id):
        if obj_id not in self.trackers.keys():
            self.trackers[obj_id] = {
                'tracker': DynamicObstacleTracker(dt=self.dt, T=self.T), # 추적 정보
                'last_update_time': rospy.Time.now()  # Initialize the last time that object updated
            }

    def initialize(self, obj_id, data):
        if obj_id in self.trackers.keys():
            self.trackers[obj_id]['tracker'].initialize(data)
            self.trackers[obj_id]['last_update_time'] = rospy.Time.now()  # 업데이트된 시간 갱신
        else:
            self.add_tracker(obj_id)
            self.trackers[obj_id]['tracker'].initialize(data)
            self.trackers[obj_id]['last_update_time'] = rospy.Time.now()  # 업데이트된 시간 갱신

    def get_deleted_ids(self):
        deleted_ids = list(self.deleted_ids)
        self.deleted_ids.clear()
        return deleted_ids

    def clean(self):
        current_time = rospy.Time.now()
        to_delete = []
        time_duration=rospy.Duration(self.timeout)
        for obj_id in self.trackers.keys():
            rospy.loginfo(f"Timeout Duration: {rospy.Duration(self.timeout)}")
            rospy.loginfo(f"Current Time: {current_time}")
            rospy.loginfo(f"Last Update Time: {self.trackers[obj_id]['last_update_time']}")
            if current_time - self.trackers[obj_id]['last_update_time'] > time_duration:
                to_delete.append(obj_id)  # 타임아웃된 객체

        for obj_id in to_delete:
            self.delete(obj_id) # 타임아웃된 객체 삭제

    # 오브젝트 관리
    def delete(self, obj_id):
        if obj_id in self.trackers.keys():
            del self.trackers[obj_id]
        else:
            print(f"obj_id")

    def update(self, obj_id, data):
        if obj_id in self.trackers.keys():
            current_time = rospy.Time.now()  # 현재 시간
            if current_time - self.trackers[obj_id]['last_update_time'] > rospy.Duration(self.timeout):
                self.delete(obj_id)  # 타임아웃된 객체 삭제
                pass

            self.trackers[obj_id]['tracker'].update(data)
            self.trackers[obj_id]['last_update_time'] = rospy.Time.now()  # 업데이트된 시간 갱신
        else:
            self.initialize(obj_id, data)

    def predict(self):
        trajs = {}
        for obj_id in self.trackers.keys():
            trajs[obj_id] = self.trackers[obj_id]['tracker'].predict()

        if len(trajs) == 0:
            return None
        else:
            return trajs


class DynamicObstacleTrackerNode:
    def __init__(self):
        rospy.init_node("dynamic_obstacle_tracker_node", anonymous=True)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_info_callback)
        self.object_pose_pub = rospy.Publisher('/Object_topic/tracked_object_pose_topic', TrackedObjectPoseList, queue_size=10)
        self.object_path_pub = rospy.Publisher('/Object_topic/tracked_object_path_topic', PredictedObjectPathList, queue_size=10)
        self.deleted_id_pub = rospy.Publisher('/Object_topic/deleted_object_id', Int32, queue_size=10)
        self.tracker = MultiDynamicObstacleTracker(dt=0.1, T=0.5, timeout=0.5)

        self.rate = rospy.Rate(30)
        self.is_object = False
        self.object_data = None
        self.previous_heading = {}
        self.previous_time = rospy.Time.now()

    def object_info_callback(self, msg):
        rospy.loginfo("Received Message")
        self.is_object=True
        self.object_data = msg

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

    def publish_object_pose(self):
        pose_list = TrackedObjectPoseList()
        pose_list.header.stamp = rospy.Time.now()

        for obj_id in self.tracker.trackers.keys():
            # [x, y, v, a, theta, theta_rate]
            x = self.tracker.trackers[obj_id]['tracker'].X[-1]
            print("Current state X:", x)
            # pose 정의, 메세지 타입은 TrackdPoint에서 x, y, v, a, theta(yaw), theta_rate(yaw_rate)로 정의
            # 이를 바탕으로 TrackedObjectPose 정의, 이를 다시 pose에 정의. pose list 반환
            pose = TrackedObjectPose(unique_id=obj_id, pose=TrackedPoint(x=x[0], y=x[1], v=x[2], a=x[3], theta=x[4], theta_rate=x[5]))
            pose_list.pose_list.append(pose)

        self.object_pose_pub.publish(pose_list)

    def publish_object_path(self):
        trajs = self.tracker.predict()
        if trajs is None:
            rospy.loginfo("No trajectories available to publish")
            return

        path_list = PredictedObjectPathList()
        path_list.header.stamp = rospy.Time.now()

        for obj_id in trajs.keys():
            path = PredictedObjectPath()
            path.unique_id = obj_id
            for point in trajs[obj_id]:
                # [x, y, v, a, theta, theta_rate]
                pose = TrackedPoint(x=point[0], y=point[1], v=point[2], a=point[3], theta=point[4], theta_rate=point[5])
                path.path.append(pose)

            path_list.path_list.append(path)
            rospy.loginfo("Publishing path for object ID: {}".format(obj_id))


        self.object_path_pub.publish(path_list)
        rospy.loginfo("Path data published")
    def publish_deleted_ids(self):
        deleted_ids = self.tracker.get_deleted_ids()
        for deleted_id in deleted_ids:
            self.deleted_id_pub.publish(deleted_id)
            del self.previous_heading[deleted_id]

    def run(self):
        while not rospy.is_shutdown():
            if self.is_object == True :
                current_time = rospy.Time.now() #타임 객체
                delta_time = (current_time - self.previous_time).to_sec()

                for obstacle in self.object_data.npc_list:
                    obj_id = obstacle.unique_id
                    data = self.data_preprocessing(obstacle, delta_time)

                    self.tracker.update(obj_id, data)

                    self.previous_heading[obj_id] = radians(obstacle.heading)

                self.previous_time = current_time # 타임 객체

                self.tracker.clean()
                self.publish_deleted_ids()
                self.publish_object_pose()
                self.publish_object_path()

            self.rate.sleep()

class DynamicObstacleTrackerHMM:
    def __init__(self):
        rospy.init_node("dynamic_obstacle_tracker_node", anonymous=True)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_info_callback)
        self.object_pose_pub = rospy.Publisher('/Object_topic/tracked_object_pose_topic', TrackedObjectPoseList, queue_size=10)
        self.object_path_pub = rospy.Publisher('/Object_topic/tracked_object_path_topic', PredictedObjectPathList, queue_size=10)
        self.deleted_id_pub = rospy.Publisher('/Object_topic/deleted_object_id', Int32, queue_size=10)
        self.rate = rospy.Rate(30)
        self.is_object = False
        self.object_data = None
        self.previous_heading = {}
        self.previous_time=rospy.Time.now()

        self.model=hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
        self.states=['Lane Keeping', 'Lane Changing']

    def train_hmm(self, data, lengths):
        self.model.fit(data, lengths)

    def predict_hmm(self, data):
        hidden_states=self.model.predict(data.reshape(-1,1))
        return hidden_states

    def object_info_callback(self, msg):
        self.is_object=True
        self.object_data=msg

if __name__ == '__main__':
    try:
        tracker = DynamicObstacleTrackerNode()
        tracker.run()
    except rospy.ROSInterruptException:
        pass
