#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import sin, cos, atan2, sqrt, pi
import numpy as np
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList
import matplotlib.pyplot as plt
import time


class pid_feedforward:
    def __init__(self):
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/lattice_path", Path, self.path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_callback)

        self.is_path = False
        self.is_odom = False
        self.is_status = False
        self.is_global_path = False

        self.forward_point = Point()
        self.current_postion = Point()

        self.vehicle_length = 5.205  # Hyundai Ioniq (hev)
        self.target_velocity = 40

        self.dt = 0.05
        self.Kp = 0.001
        self.Kd = 0.001
        self.Ki = 0.001
        self.kff = 0.001
        self.error = 0.0
        self.error_prev = self.error
        self.error_d = 0.0
        self.error_i = 0.0
        self.max_delta_error = 3.0
        self.u = 0
        self.feedforwardterm = 0
        self.coeff = None

        self.path = None
        self.x_ego = []
        self.y_ego = []
        self.start_time = None
        self.end_time = None
        self.errors = []

        self.lookahead_distance = 20  # Lookahead distance 설정

    def global_path_callback(self, msg):
        self.global_path = msg
        self.is_global_path = True

    def object_callback(self, msg):
        self.is_obj = True
        self.object_data = msg

    def path_callback(self, msg):
        self.is_path = True
        self.path = msg

        x = []
        y = []

        for pose in msg.poses:
            global_position = pose.pose.position
            local_position = self.transform_to_local(global_position, self.current_postion, self.vehicle_yaw)
            x.append(local_position.x)
            y.append(local_position.y)

        if len(x) > 3:
            self.coeff = np.polyfit(x, y, 3)
            self.coeff = self.coeff[::-1].reshape(-1, 1)
        else:
            self.coeff = None

    def odom_callback(self, msg):
        self.is_odom = True
        odom_quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                           msg.pose.pose.orientation.w)
        _, _, self.vehicle_yaw = euler_from_quaternion(odom_quaternion)
        self.current_postion.x = msg.pose.pose.position.x
        self.current_postion.y = msg.pose.pose.position.y
        self.x_ego.append(self.current_postion.x)
        self.y_ego.append(self.current_postion.y)

        if self.start_time is None:
            self.start_time = time.time()

    def status_callback(self, msg):
        self.is_status = True
        self.status_msg = msg


    def transform_to_local(self, global_position, reference_position, reference_theta):
        # 글로벌 좌표를 로컬 좌표로 변환
        translation = np.array([global_position.x - reference_position.x,
                                global_position.y - reference_position.y])
        rotation_matrix = np.array([[cos(reference_theta), sin(reference_theta)],
                                    [-sin(reference_theta), cos(reference_theta)]])
        local_position = rotation_matrix.dot(translation)
        return Point(x=local_position[0], y=local_position[1], z=0)

    def compute_cte(self):
        if self.path is None or not self.path.poses:
            return 0.0

        min_dist = float('inf')
        closest_idx = 0

        # 현재 위치에서 가장 가까운 점 찾기
        for i, pose in enumerate(self.path.poses):
            global_position = pose.pose.position
            local_position = self.transform_to_local(global_position, self.current_postion, self.vehicle_yaw)

            dist = sqrt(local_position.x**2 + local_position.y**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        lookahead_idx = closest_idx

        if self.is_obstacle_nearby():
            # Lookahead distance를 정확히 반영하여 다음 지점 선택
            for i in range(closest_idx, len(self.path.poses)):
                global_position = self.path.poses[i].pose.position
                local_position = self.transform_to_local(global_position, self.current_postion, self.vehicle_yaw)
                lookahead_dist = sqrt(local_position.x**2 + local_position.y**2)
                if lookahead_dist >= self.lookahead_distance:
                    lookahead_idx = i
                    break

        if lookahead_idx == closest_idx:
            lookahead_idx = min(lookahead_idx + 1, len(self.path.poses) - 1)

        # Lookahead point에서의 횡방향 거리 계산
        lookahead_point = self.path.poses[lookahead_idx].pose.position
        lookahead_local = self.transform_to_local(lookahead_point, self.current_postion, self.vehicle_yaw)

        cte = lookahead_local.y  # 횡방향 거리만 사용하여 CTE 계산

        if abs(cte) < 100:
            self.errors.append(abs(cte))

        return cte

    def calc_pid_feedforward(self):
        if not self.is_path or not self.is_odom or not self.is_status:
            return 0.0

        cte = self.compute_cte()
        if self.coeff is None:
            return 0.0
        print(cte)
        max_cte = 10.0
        cte = np.clip(cte, -max_cte, max_cte)

        if abs(cte) > 0.2:
            self.Kp = 0.05
            self.Kd = 0.03
            self.Ki = 0.001
            self.kff = 0.001
        else:
            self.Kp = 0.01
            self.Kd = 0.05
            self.Ki = 0.005
            self.kff = 0.01

        self.error = cte

        self.error_d = (self.error - self.error_prev) / self.dt
        self.error_i = self.error_i + self.error * self.dt
        self.feedforwardterm = self.status_msg.velocity.x**2 * 2 * self.coeff[-3][0]

        self.u = self.Kp * self.error + self.Kd * self.error_d + self.Ki * self.error_i + self.kff * self.feedforwardterm

        # # 조향각 제한 설정 (필요 시)
        # max_steering_rate = pi / 45
        # if abs(self.u - self.error_prev) > max_steering_rate:
        #     if self.u > self.error_prev:
        #         self.u = self.error_prev + max_steering_rate
        #     else:
        #         self.u = self.error_prev - max_steering_rate

        self.error_prev = self.error

        return self.u



    def is_obstacle_nearby(self):
        if not self.is_obj:
            return False

        for obj in self.object_data.npc_list:
            local_position = self.transform_to_local(obj.position, self.current_postion, self.vehicle_yaw)
            distance = sqrt(local_position.x**2 + local_position.y**2)
            if distance < 40 and local_position.x > -15 and abs(local_position.y) < 5 :
                return True
        return False


    def get_current_waypoint(self, ego_status, global_path):
        min_dist = float('inf')
        current_waypoint = -1
        for i, pose in enumerate(global_path.poses):
            dx = ego_status.position.x - pose.pose.position.x
            dy = ego_status.position.y - pose.pose.position.y

            dist = sqrt(pow(dx, 2) + pow(dy, 2))
            if min_dist > dist:
                min_dist = dist
                current_waypoint = i
        return current_waypoint

    def calculate_statistics(self):
        if len(self.errors) > 0:
            mean_error = np.mean(self.errors)
            max_error = np.max(self.errors)
            variance = np.var(self.errors)
            return mean_error, max_error, variance
        return 0.0, 0.0, 0.0

    def calculate_total_time(self):
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        elif self.start_time is not None:
            return time.time() - self.start_time
        return 0.0

    def set_end_time(self):
        if self.end_time is None:
            self.end_time = time.time()

def plot_paths(global_path, x_ego, y_ego, total_time, variance, mean_error, max_error):
    if global_path is None:
        return

    x_global = [pose.pose.position.x for pose in global_path.poses]
    y_global = [pose.pose.position.y for pose in global_path.poses]

    plt.figure(figsize=(10, 6))
    plt.plot(x_global, y_global, 'k--', label='Global Path')
    plt.plot(x_ego, y_ego, 'b-', label='Ego Vehicle Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
    plt.title('Vehicle Path Tracking')
    plt.grid(True)

    plt.text(0.95, 0.05, f'Total Time: {total_time:.2f}s\nVariance: {variance:.4f}\nMean Error: {mean_error:.4f}\nMax Error: {max_error:.4f}', 
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=plt.gca().transAxes)

    plt.show()

# if __name__ == "__main__":
#     rospy.init_node('path_tracking_node', anonymous=True)

#     pid_controller = pid_feedforward()

#     rate = rospy.Rate(10)  # 10 Hz
#     while not rospy.is_shutdown():
#         if pid_controller.is_path and pid_controller.is_odom and pid_controller.is_status:
#             steering = pid_controller.calc_pid_feedforward()
#         rate.sleep()

#     pid_controller.set_end_time()

#     if pid_controller.global_path is not None:
#         mean_error, max_error, variance = pid_controller.calculate_statistics()
#         plot_paths(pid_controller.global_path, pid_controller.x_ego, pid_controller.y_ego,
#                    pid_controller.calculate_total_time(), variance, mean_error, max_error)
