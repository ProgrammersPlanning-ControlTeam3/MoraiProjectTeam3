#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import cos, sin, sqrt, pow, atan2, pi
import numpy as np
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from morai_msgs.msg import EgoVehicleStatus
import matplotlib.pyplot as plt
import time


class pure_pursuit_no_npc:
    def __init__(self):
        rospy.Subscriber("/local_path", Path, self.path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)

        self.is_path = False
        self.is_odom = False
        self.is_status = False

        self.forward_point = Point()
        self.current_postion = Point()

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
        self.x_ego.append(self.current_postion.x)
        self.y_ego.append(self.current_postion.y)

        if self.start_time is None:
            self.start_time = time.time()

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
        if not self.is_path or not self.is_status:
            return 0.0

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

class pure_pursuit:
    def __init__(self):
        rospy.Subscriber("/lattice_path", Path, self.path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)

        self.is_path = False
        self.is_odom = False
        self.is_status = False

        self.forward_point = Point()
        self.current_postion = Point()

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
        self.x_ego.append(self.current_postion.x)
        self.y_ego.append(self.current_postion.y)

        if self.start_time is None:
            self.start_time = time.time()

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
        if not self.is_path or not self.is_status:
            return 0.0

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

class stanley:
    def __init__(self):
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/lattice_path", Path, self.path_callback)
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

        self.k = 1.5  # Stanley Gain
        self.k_psi = 1.0  # For heading Error
        self.k_y = 4.0  # For CTR Error

        # self.k = 1.1  # Stanley Gain
        # self.k_psi = 0.8  # For heading Error
        # self.k_y = 0.75  # For CTR Error

        self.max_cross_track_error = 0.4  # Maximum cross track error
        self.alpha = 9

        self.vehicle_length = 5.155  # Vehicle Length
        self.lfd = 10  # Look forward Distance
        if self.vehicle_length is None or self.lfd is None:
            print("you need to change values at line 57~58 ,  self.vegicle_length , lfd")
            exit()

        self.global_path = None
        self.x_ego = []
        self.y_ego = []
        self.start_time = None
        self.end_time = None
        self.errors = []

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
        self.x_ego.append(self.current_position.x)
        self.y_ego.append(self.current_position.y)

        if self.start_time is None:
            self.start_time = time.time()

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

            dist = sqrt(dx ** 2 + dy ** 2)
            if min_dist > dist:
                min_dist = dist
                current_waypoint = i
        return current_waypoint

    def calc_stanley_control(self):
        if not self.is_path or not self.is_odom:
            return 0.0  # No control if path or odom is not available

        min_distance = float('inf')

        # Cross track error
        path_point = self.path.poses[0].pose.position
        dx = self.current_position.x - path_point.x
        dy = self.current_position.y - path_point.y
        cos_yaw = cos(self.vehicle_yaw)
        sin_yaw = sin(self.vehicle_yaw)
        cross_track_error = dx * sin_yaw - dy * cos_yaw

        if abs(cross_track_error) < 100:  # 이상치 제거를 위한 조건 추가
            self.errors.append(abs(cross_track_error))

        cross_track_error = np.clip(cross_track_error, -self.max_cross_track_error, self.max_cross_track_error)

        path_point_local_x = cos(self.vehicle_yaw) * (path_point.x - self.current_position.x) + sin(self.vehicle_yaw) * (path_point.y - self.current_position.y)
        path_point_local_y = -sin(self.vehicle_yaw) * (path_point.x - self.current_position.x) + cos(self.vehicle_yaw) * (path_point.y - self.current_position.y)

        if path_point_local_y > 0:
            cross_track_error = abs(cross_track_error)
        else:
            cross_track_error = -abs(cross_track_error)

        lookahead_idx = 8  # Look ahead points
        if lookahead_idx >= len(self.path.poses):
            lookahead_idx = len(self.path.poses) - 1
        dx = self.path.poses[lookahead_idx].pose.position.x - self.path.poses[0].pose.position.x
        dy = self.path.poses[lookahead_idx].pose.position.y - self.path.poses[0].pose.position.y
        path_heading = atan2(dy, dx)
        heading_error = path_heading - self.vehicle_yaw

        # Normalize heading error to the range [-pi, pi]
        while heading_error > pi:
            heading_error -= 2 * pi
        while heading_error < -pi:
            heading_error += 2 * pi

        # Speed-dependent factor (alpha)
        alpha = self.alpha / max(self.status_msg.velocity.x, 0.1)  # Prevent division by zero

        # Calculate steering using Stanley method
        CTR = atan2(self.k * cross_track_error, self.target_velocity)

        # Desired input with speed-dependent heading error
        steering = (self.k_psi * heading_error * alpha) + (self.k_y * CTR)

        return steering


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
        rospy.logwarn("Global path is None, cannot plot paths.")
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

    # Display total time, variance, mean error, and max error in the plot
    plt.text(0.95, 0.05, f'Total Time: {total_time:.2f}s\nVariance: {variance:.4f}\nMean Error: {mean_error:.4f}\nMax Error: {max_error:.4f}', 
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=plt.gca().transAxes)

    plt.show()

if __name__ == "__main__":
    rospy.init_node('path_tracking_node', anonymous=True)

    pure_pursuit_controller = pure_pursuit_no_npc()
    stanley_controller = stanley()

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if pure_pursuit_controller.is_path and pure_pursuit_controller.is_odom and pure_pursuit_controller.is_status:
            pure_pursuit_controller.calc_pure_pursuit()
        if stanley_controller.is_path and stanley_controller.is_odom and stanley_controller.is_status:
            stanley_controller.calc_stanley_control()
        rate.sleep()

    # End the simulation by setting the end time
    pure_pursuit_controller.set_end_time()
    stanley_controller.set_end_time()

    if pure_pursuit_controller.path is not None:
        mean_error, max_error, variance = pure_pursuit_controller.calculate_statistics()
        plot_paths(pure_pursuit_controller.path, pure_pursuit_controller.x_ego, pure_pursuit_controller.y_ego,
                   pure_pursuit_controller.calculate_total_time(), variance, mean_error, max_error)

    if stanley_controller.global_path is not None:
        mean_error, max_error, variance = stanley_controller.calculate_statistics()
        plot_paths(stanley_controller.global_path, stanley_controller.x_ego, stanley_controller.y_ego,
                   stanley_controller.calculate_total_time(), variance, mean_error, max_error)

