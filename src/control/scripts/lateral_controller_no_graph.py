#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import cos, sin, pi, sqrt, pow, atan2
import numpy as np
from tf.transformations import euler_from_quaternion

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from morai_msgs.msg import EgoVehicleStatus


class pure_pursuit_no_npc:
    def __init__(self):

        # rospy.init_node('pure_pursuit', anonymous=True)

        rospy.Subscriber("/local_path", Path, self.path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)

        self.is_path = False
        self.is_odom = False
        self.is_status = False

        self.forward_point = Point()
        self.current_postion = Point()

        self.vehicle_length = 5.205  # Hyeondai Ioniq (hev)
        self.lfd = 10
        self.min_lfd = 10
        self.max_lfd = 30  # default 30
        self.lfd_gain = 0.78  # default 0.78

    def path_callback(self, msg):
        self.is_path = True
        self.path = msg

    def odom_callback(self, msg):
        self.is_odom = True
        odom_quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                           msg.pose.pose.orientation.w)
        _, _, self.vehicle_yaw = euler_from_quaternion(odom_quaternion)
        self.current_postion.x = msg.pose.pose.position.x
        self.current_postion.y = msg.pose.pose.position.y

    def status_callback(self, msg):  ## Vehicle Status Subscriber
        self.is_status = True
        self.status_msg = msg

    def get_current_waypoint(self, ego_status, global_path):
        min_dist = float('inf')
        currnet_waypoint = -1
        for i, pose in enumerate(global_path.poses):
            dx = ego_status.position.x - pose.pose.position.x
            dy = ego_status.position.y - pose.pose.position.y

            dist = sqrt(pow(dx, 2) + pow(dy, 2))
            if min_dist > dist:
                min_dist = dist
                currnet_waypoint = i
        return currnet_waypoint

    def calc_pure_pursuit(self):
        local_path_point = None

        self.lfd = (self.status_msg.velocity.x) * self.lfd_gain

        if self.lfd < self.min_lfd:
            self.lfd = self.min_lfd
        elif self.lfd > self.max_lfd:
            self.lfd = self.max_lfd

        vehicle_position = self.current_postion
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
                if dis >= self.lfd:
                    self.forward_point = path_point
                    self.is_look_forward_point = True
                    break

        if not self.is_look_forward_point:
            rospy.logwarn("No forward point found, unable to compute steering.")
            return 0.0

        theta = atan2(local_path_point[1], local_path_point[0])
        steering = atan2((2 * self.vehicle_length * sin(theta)), self.lfd)
        # print(steering)
        return steering

class pure_pursuit:
    def __init__(self):


        # rospy.init_node('pure_pursuit', anonymous=True)

        rospy.Subscriber("/lattice_path", Path, self.path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)

        self.is_path = False
        self.is_odom = False
        self.is_status = False

        self.forward_point = Point()
        self.current_postion = Point()

        self.vehicle_length = 5.205  # Hyeondai Ioniq (hev)
        self.lfd = 10
        self.min_lfd = 10
        self.max_lfd = 30  # default 30
        self.lfd_gain = 0.78  # default 0.78

    def path_callback(self, msg):
        self.is_path = True
        self.path = msg

    def odom_callback(self, msg):
        self.is_odom = True
        odom_quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                           msg.pose.pose.orientation.w)
        _, _, self.vehicle_yaw = euler_from_quaternion(odom_quaternion)
        self.current_postion.x = msg.pose.pose.position.x
        self.current_postion.y = msg.pose.pose.position.y

    def status_callback(self, msg):  ## Vehicle Status Subscriber
        self.is_status = True
        self.status_msg = msg
        self.global_path = msg
        self.is_global_path = True

    def get_current_waypoint(self, ego_status, global_path):
        min_dist = float('inf')
        currnet_waypoint = -1
        for i, pose in enumerate(global_path.poses):
            dx = ego_status.position.x - pose.pose.position.x
            dy = ego_status.position.y - pose.pose.position.y

            dist = sqrt(pow(dx, 2) + pow(dy, 2))
            if min_dist > dist:
                min_dist = dist
                currnet_waypoint = i
        return currnet_waypoint

    def calc_pure_pursuit(self):
        local_path_point = None

        self.lfd = (self.status_msg.velocity.x) * self.lfd_gain

        if self.lfd < self.min_lfd:
            self.lfd = self.min_lfd
        elif self.lfd > self.max_lfd:
            self.lfd = self.max_lfd
        vehicle_position = self.current_postion
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
                if dis >= self.lfd:
                    self.forward_point = path_point
                    self.is_look_forward_point = True
                    break

        if local_path_point is None:
            rospy.logwarn("No valid local path point found")
            return 0.0  # 기본값 반환 또는 이전 스티어링 값

        theta = atan2(local_path_point[1], local_path_point[0])
        steering = atan2((2 * self.vehicle_length * sin(theta)), self.lfd)
        # print(steering)
        return steering

class stanley:
    def __init__(self):
        # rospy.init_node('stanley', anonymous=True)

        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/local_path", Path, self.path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)

        self.is_path = False
        self.is_odom = False
        self.is_status = False
        self.is_global_path = False

        self.is_look_forward_point = True

        self.forward_point = Point()
        self.current_position = Point()

        self.target_velocity = 40  # Target Velocity in m/s

        self.k = 1.4  # Stanley Gain
        self.k_psi = 0.8  # For heading Error
        self.k_y = 1  # For CTR Error

        self.max_cross_track_error = 1.0  # Maximum cross track error


        self.vehicle_length = 5.155  # Vehicle Length
        self.lfd = 10  # Look forward Distance
        if self.vehicle_length is None or self.lfd is None:
            print("you need to change values at line 57~58 ,  self.vegicle_length , lfd")
            exit()

    def path_callback(self, msg):
        self.is_path = True
        self.path = msg

    def odom_callback(self, msg):
        self.is_odom = True
        odom_quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        _, _, self.vehicle_yaw = euler_from_quaternion(odom_quaternion)
        self.current_position.x = msg.pose.pose.position.x
        self.current_position.y = msg.pose.pose.position.y

    def status_callback(self, msg):  ## Vehicle Status Subscriber
        self.is_status = True
        self.status_msg = msg

    def global_path_callback(self, msg):
        self.global_path = msg
        self.is_global_path = True

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

    def calc_stanley_control(self):
        if not self.is_global_path or not self.is_odom:
            return 0.0  # No control if path or odom is not available

        min_distance = float('inf')
        nearest_idx = -0.0

            # Find the point on the path closest to the vehicle
        for i, pose in enumerate(self.global_path.poses):
            dx = self.current_position.x - pose.pose.position.x
            dy = self.current_position.y - pose.pose.position.y
            distance = sqrt(dx ** 2 + dy ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_idx = i

        # Cross track error
        nearest_point = self.global_path.poses[nearest_idx].pose.position
        dx = self.current_position.x - nearest_point.x
        dy = self.current_position.y - nearest_point.y
        cos_yaw = cos(self.vehicle_yaw)
        sin_yaw = sin(self.vehicle_yaw)
        cross_track_error = dx * sin_yaw - dy * cos_yaw

        cross_track_error = np.clip(cross_track_error, -self.max_cross_track_error, self.max_cross_track_error)

        path_point = self.global_path.poses[nearest_idx].pose.position
        path_point_local_x = cos(self.vehicle_yaw) * (path_point.x - self.current_position.x) + sin(self.vehicle_yaw) * (path_point.y - self.current_position.y)
        path_point_local_y = -sin(self.vehicle_yaw) * (path_point.x - self.current_position.x) + cos(self.vehicle_yaw) * (path_point.y - self.current_position.y)

        if path_point_local_y > 0:
            cross_track_error = abs(cross_track_error)
        else:
            cross_track_error = -abs(cross_track_error)

        lookahead_idx = nearest_idx + 8  # Look ahead points
        if lookahead_idx >= len(self.global_path.poses):
            lookahead_idx = len(self.global_path.poses) - 1
        dx = self.global_path.poses[lookahead_idx].pose.position.x - self.global_path.poses[nearest_idx].pose.position.x
        dy = self.global_path.poses[lookahead_idx].pose.position.y - self.global_path.poses[nearest_idx].pose.position.y
        path_heading = atan2(dy, dx)
        heading_error = path_heading - self.vehicle_yaw

        # Normalize heading error to the range [-pi, pi]
        while heading_error > pi:
            heading_error -= 2 * pi
        while heading_error < -pi:
            heading_error += 2 * pi

        # Calculate steering using Stanley method
        CTR = atan2(self.k * cross_track_error, self.target_velocity)

        # Desired input
        steering = self.k_psi * heading_error + self.k_y * CTR

        return steering