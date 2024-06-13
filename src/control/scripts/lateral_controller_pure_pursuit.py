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
from control.scripts.controller_utils import unified_calculator, plot_paths, get_waypoint

class pure_pursuit:
    def __init__(self):
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/lattice_path", Path, self.path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)

        self.is_path = False
        self.is_odom = False
        self.is_status = False
        self.is_global_path = False

        self.forward_point = Point()
        self.current_position = Point()

        self.vehicle_length = 5.205  # Hyundai Ioniq (hev)
        self.lfd = 10
        self.min_lfd = 10
        self.max_lfd = 30  # default 30
        self.lfd_gain = 0.78  # default 0.78

        self.path = None
        self.x_ego = []
        self.y_ego = []
        self.start_time = None
        self.end_time = None
        self.errors = []


    def global_path_callback(self, msg):
        self.global_path = msg
        self.is_global_path = True


    def path_callback(self, msg):
        self.is_path = True
        self.path = msg

    def odom_callback(self, msg):
        self.is_odom = True
        odom_quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                           msg.pose.pose.orientation.w)
        _, _, self.vehicle_yaw = euler_from_quaternion(odom_quaternion)
        self.current_position.x = msg.pose.pose.position.x
        self.current_position.y = msg.pose.pose.position.y
        self.x_ego.append(self.current_position.x)
        self.y_ego.append(self.current_position.y)

        if self.start_time is None:
            self.start_time = time.time()

    def status_callback(self, msg):  ## Vehicle Status Subscriber
        self.is_status = True
        self.status_msg = msg


    def calc_pure_pursuit(self):
        if not self.is_path or not self.is_status:
            return 0.0

        local_path_point = None
        self.lfd = (self.status_msg.velocity.x) * self.lfd_gain

        if self.lfd < self.min_lfd:
            self.lfd = self.min_lfd
        elif self.lfd > self.max_lfd:
            self.lfd = self.max_lfd
        vehicle_position = self.current_position
        self.is_look_forward_point = False

        translation = [vehicle_position.x, vehicle_position.y]

        trans_matrix = np.array([
            [cos(self.vehicle_yaw), -sin(self.vehicle_yaw), translation[0]],
            [sin(self.vehicle_yaw), cos(self.vehicle_yaw), translation[1]],
            [0, 0, 1]])

        det_trans_matrix = np.linalg.inv(trans_matrix)

        for num, i in enumerate(self.path.poses):
            path_point = i.pose.position

            global_path_point = [path_point.x, path_point.y, 1]
            local_path_point = det_trans_matrix.dot(global_path_point)

            if local_path_point[0] > 0:
                dis = sqrt(pow(local_path_point[0], 2) + pow(local_path_point[1], 2))
                if dis < 100:  # 이상치 제거를 위한 조건 추가
                    self.errors.append(dis)
                if dis >= self.lfd:
                    self.forward_point = path_point
                    self.is_look_forward_point = True
                    break

        if not self.is_look_forward_point:
            rospy.logwarn("No forward point found, unable to compute steering.")
            return 0.0

        theta = atan2(local_path_point[1], local_path_point[0])
        steering = atan2((2 * self.vehicle_length * sin(theta)), self.lfd)
        return steering


    def get_current_waypoint(self, ego_status, global_path):
        return get_waypoint(ego_status, global_path)


    def perform_calculations(self):
        statistics = unified_calculator(errors=self.errors, operation='statistics')
        total_time = unified_calculator(start_time=self.start_time, end_time=self.end_time, operation='total_time')
        self.end_time = unified_calculator(end_time=self.end_time, operation='set_end_time')

        return statistics, total_time





if __name__ == "__main__":
    rospy.init_node('path_tracking_node', anonymous=True)

    pure_pursuit_controller = pure_pursuit()

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if pure_pursuit_controller.is_path and pure_pursuit_controller.is_odom and pure_pursuit_controller.is_status:
            pure_pursuit_controller.calc_pure_pursuit()
        rate.sleep()

    # End the simulation by setting the end time and perform calculations
    statistics, total_time = pure_pursuit_controller.perform_calculations()

    if pure_pursuit_controller.global_path is not None:
        mean_error, max_error, variance = statistics
        plot_paths(pure_pursuit_controller.global_path, pure_pursuit_controller.x_ego, pure_pursuit_controller.y_ego,
                   total_time, variance, mean_error, max_error)
