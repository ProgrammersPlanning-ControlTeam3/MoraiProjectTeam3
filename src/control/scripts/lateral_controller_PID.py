#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import cos, sin, sqrt, pow, atan2, pi
import numpy as np
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, '/home/ubuntu/MoraiProjectTeam3/src')
from control.scripts.controller_utils import unified_calculator, plot_paths, get_waypoint, is_obstacle_nearby

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

        self.lookahead_distance = 20

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


    def global_to_local(self, global_path, current_position, current_yaw):
        local_path = []
        for pose in global_path.poses:
            dx = pose.pose.position.x - current_position.x
            dy = pose.pose.position.y - current_position.y

            local_x = dx * cos(-current_yaw) - dy * sin(-current_yaw)
            local_y = dx * sin(-current_yaw) + dy * cos(-current_yaw)

            local_path.append((local_x, local_y))
        return local_path


    def compute_cte(self):
        if self.path is None or not self.path.poses:
            return 0.0

        min_dist = float('inf')
        closest_idx = 0

        for i, pose in enumerate(self.path.poses):
            global_position = pose.pose.position
            local_position = self.transform_to_local(global_position, self.current_postion, self.vehicle_yaw)

            dist = sqrt(local_position.x**2 + local_position.y**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        lookahead_idx = closest_idx
        # print(len(self.path.poses))

        if self.is_obstacle_nearby():
            for i in range(closest_idx, len(self.path.poses)):
                global_position = self.path.poses[i].pose.position
                local_position = self.transform_to_local(global_position, self.current_postion, self.vehicle_yaw)
                lookahead_dist = sqrt(local_position.x**2 + local_position.y**2)
                if lookahead_dist >= self.lookahead_distance:
                    lookahead_idx = i
                    break

        if lookahead_idx == closest_idx:
            lookahead_idx = min(lookahead_idx + 1, len(self.path.poses) - 1)

        lookahead_point = self.path.poses[lookahead_idx].pose.position
        lookahead_local = self.transform_to_local(lookahead_point, self.current_postion, self.vehicle_yaw)

        cte = lookahead_local.y

        if abs(cte) < 100:
            self.errors.append(abs(cte))

        return cte


    def calc_pid_feedforward(self):
        if not self.is_path or not self.is_odom or not self.is_status:
            return 0.0

        cte = self.compute_cte()
        # print(cte)
        if self.coeff is None:
            return 0.0
        max_cte = 10.0
        cte = np.clip(cte, -max_cte, max_cte)

        if abs(cte) > 0.3:
            self.Kp = 0.05
            self.Kd = 0.001
            self.Ki = 0.00
            self.kff = 0.0001
        else:
            self.Kp = 0.6
            self.Kd = 0.02
            self.Ki = 0.001
            self.kff = 0.001

        self.error = cte

        self.error_d = (self.error - self.error_prev) / self.dt
        self.error_i = self.error_i + self.error * self.dt
        self.feedforwardterm = self.status_msg.velocity.x**2 * 2 * self.coeff[-3][0]

        self.u = self.Kp * self.error + self.Kd * self.error_d + self.Ki * self.error_i + self.kff * self.feedforwardterm

        # if abs(cte) > 0.3:
            # 조향각 제한 설정 (필요 시)
            # max_steering_rate = pi / 3600

            # if abs(self.u - self.error_prev) > 0.5:
            #     if self.u > self.error_prev:
            #         self.u = self.error_prev + max_steering_rate
            #         print("+")
            #     else:
            #         self.u = self.error_prev - max_steering_rate
            #         print("-")
            # print(self.u)
        self.error_prev = self.error

        max_steering_angle = pi / 18
        self.u = np.clip(self.u, -max_steering_angle, max_steering_angle)

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
        return get_waypoint(ego_status, global_path)


    def perform_calculations(self):
        statistics = unified_calculator(errors=self.errors, operation='statistics')
        total_time = unified_calculator(start_time=self.start_time, end_time=self.end_time, operation='total_time')
        self.end_time = unified_calculator(end_time=self.end_time, operation='set_end_time')

        return statistics, total_time




if __name__ == "__main__":
    rospy.init_node('path_tracking_node', anonymous=True)

    pid_controller = pid_feedforward()

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if pid_controller.is_path and pid_controller.is_odom and pid_controller.is_status:
            steering = pid_controller.calc_pid_feedforward()
        rate.sleep()

    # End the simulation by setting the end time and perform calculations
    statistics, total_time = pid_controller.perform_calculations()

    if pid_controller.global_path is not None:
        mean_error, max_error, variance = statistics
        plot_paths(pid_controller.global_path, pid_controller.x_ego, pid_controller.y_ego,
                   total_time, variance, mean_error, max_error)
