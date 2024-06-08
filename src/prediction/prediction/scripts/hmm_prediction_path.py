#!/usr/bin/env python3
# coding: utf-8
import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Path
from prediction.msg import TrackedPoint, PredictedObjectPath, PredictedObjectPathList, TrackedObjectPose, TrackedObjectPoseList, PredictedHMM, ObjectFrenetPosition
from morai_msgs.msg import ObjectStatusList
from utils import *
class HMMPredictionPath:
    def __init__(self):
        rospy.init_node("/hmm_prediction_path_node", anonymous=True)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_info_callback)
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/Object_topic/hmm_prediction", PredictedHMM, self.prediction_info_callback)
        rospy.Subscriber("/Object_topic/frenet_position", ObjectFrenetPosition, self.object_frenet_info_callback)
        self.hmm_based_path = rospy.Publisher('/Object_topic/hmm_predicted_path_topic', PredictedObjectPathList, queue_size=10)
        self.rate = rospy.Rate(30)
        self.is_prediction_received = False
        self.prediction_data = None
        self.is_object = False
        self.is_global_path = False
        self.is_frenet_data = False
        self.frenet_data = None

    def obejct_frenet_info_callback(self, msg):
        self.is_frenet_data = True
        self.frenet_data = msg

    def global_path_callback(self, msg):
        self.global_path_msg = msg
        self.is_global_path = True

    def object_info_callback(self, msg):
        rospy.loginfo("Received Object info Message")
        self.is_object= True
        self.object_data = msg

    def prediction_info_callback(self, msg):
        rospy.loginfo("Received prediction msg data")
        self.is_prediction_received = True
        self.prediction_data = msg

    def generate_polynomial_path(self, start, end, order):
        # start and end is a type of tuple (x, y, dy/dx)
        x0, y0, dy0 = start
        x1, y1, dy1 = end

        if order == 3:
            # 3차 다항식: y = ax^3 + bx^2 + cx + d
            A = np.array([
                [x0**3, x0**2, x0, 1],
                [x1**3, x1**2, x1, 1],
                [3*x0**2, 2*x0, 1, 0],
                [3*x1**2, 2*x1, 1, 0]
            ])
            b = np.array([y0, y1, dy0, dy1])
        elif order == 5:
            # 5차 다항식: y = ax^5 + bx^4 + cx^3 + dx^2 + ex + f
            A = np.array([
                [x0**5, x0**4, x0**3, x0**2, x0, 1],
                [x1**5, x1**4, x1**3, x1**2, x1, 1],
                [5*x0**4, 4*x0**3, 3*x0**2, 2*x0, 1, 0],
                [5*x1**4, 4*x1**3, 3*x1**2, 2*x1, 1, 0],
                [20*x0**3, 12*x0**2, 6*x0, 2, 0, 0],
                [20*x1**3, 12*x1**2, 6*x1, 2, 0, 0]
            ])
            b = np.array([y0, y1, dy0, dy1, 0, 0])  # 가속도 0으로 가정
        coefficients = np.linalg.solve(A, b)
        return coefficients

    def evaluate_polynomial(self, coefficients, x_range):
        """다항식 계수와 x의 범위를 받아 y의 값을 계산합니다."""
        return np.polyval(coefficients[::-1], x_range)
    def create_lane_change_path(self, current_x, current_y, current_v, predicted_state, lane_width=3.521):
        # 경로 생성 시뮬레이션 범위
        x_range = np.linspace(current_x, current_x + 100, num=100)  # 100m 앞까지 예측

        if predicted_state == "Right Lane Change":
            # 오른쪽 차선 변경
            end_y = current_y + lane_width
            coefficients = self.generate_polynomial_path((current_x, current_y, current_v), (current_x + 50, end_y, current_v))
        elif predicted_state == "Left Lane Change":
            # 왼쪽 차선 변경
            end_y = current_y - lane_width
            coefficients = self.generate_polynomial_path((current_x, current_y, current_v), (current_x + 50, end_y, current_v))
        else:
            # 차선 유지
            coefficients = self.generate_polynomial_path((current_x, current_y, current_v), (current_x + 50, current_y, current_v))

        path_y = self.evaluate_polynomial(coefficients, x_range)
        return x_range, path_y


    def path_publisher(self):
        if self.is_prediction_received and self.is_frenet_data:
            predicted_paths = PredictedObjectPathList()
            predicted_paths.header.stamp = rospy.Time.now()
            predicted_paths.header.frame_id = "map"

            for prediction in self.prediction_data.predictions:
                for frenet_position in self.frenet_data:
                    if prediction.unique_id == frenet_position.unique_id:
                        s_range, d_path = self.create_lane_change_path(
                            frenet_position.s, frenet_position.d, frenet_position.speed, prediction.maneuver
                        )
                        path_msg = PredictedObjectPath()
                        path_msg.header.stamp = rospy.Time.now()
                        path_msg.header.framed_id = "map"
                        path_msg.unique_id = prediction.unique_id
                        path_msg.path = []

                        for s, d in zip(s_range, d_path):
                            mapx = [pose.pose.position.x for pose in self.global_path_msg.poses]
                            mapy = [pose.pose.position.y for pose in self.global_path_msg.poses]
                            maps =[0]
                            for i in range(1, len(mapx)):
                                maps.append(maps[-1] + get_dist(mapx[i-1], mapy[i-1], mapx[i], mapy[i]))
                            x, y =get_cartesian(s, d, mapx, mapy, maps)  #### maps have to be defined.... HOW?
                            point = TrackedPoint()
                            point.x = x
                            point.y=y
                            path_msg.path.append(point)
                        predicted_paths.paths.append(path_msg)
            self.hmm_based_path.publish(predicted_paths)
            rospy.loginfo("Pubslihed predicted paths for all vehicles.")

if __name__ == "__main__":
    try:
        paths = HMMPredictionPath()
        paths.path_publisher()
    except rospy.ROSInitException:
        pass