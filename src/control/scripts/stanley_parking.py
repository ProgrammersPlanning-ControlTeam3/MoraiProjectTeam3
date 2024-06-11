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



class stanley_parking:
    def __init__(self):
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/lattice_path", Path, self.path_callback)
        rospy.Subscriber("/local_path", Path, self.local_path_callback)

        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_callback)

        self.is_path = False
        self.is_odom = False
        self.is_status = False
        self.is_global_path = False

        self.forward_point = Point()
        self.current_position = Point()

        self.target_velocity = 40  # Target Velocity in m/s

        self.k = 0.5  # Stanley Gain
        self.k_psi = 0.5  # For heading Error
        self.k_y = 1.0  # For CTR Error

        self.max_cross_track_error = 0.4  # Maximum cross track error
        self.alpha = 8

        self.vehicle_length = 5.155  # Vehicle Length


        self.global_path = None
        self.x_ego = []
        self.y_ego = []
        self.start_time = None
        self.end_time = None
        self.errors = []

    def path_callback(self, msg):
        self.is_path = True
        self.path = msg

    def local_path_callback(self, msg):
        self.is_local_path = True
        self.local_path = msg

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

    def object_callback(self, msg):
        self.is_obj = True
        self.object_data = msg

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

    def calc_stanley_control_local(self):
        if not self.is_path or not self.is_odom or not self.is_status:
            return 0.0

        current_velocity = self.status_msg.velocity.x

        # 가장 가까운 포인트 찾기
        closest_distance = float('inf')
        closest_point = None
        for pose in self.path.poses:
            path_point = pose.pose.position
            distance = sqrt((path_point.x - self.current_position.x) ** 2 +
                            (path_point.y - self.current_position.y) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_point = path_point

        if closest_point is None:
            return 0.0

        # CTE (Cross Track Error) 계산
        dx = closest_point.x - self.current_position.x
        dy = closest_point.y - self.current_position.y
        cos_yaw = cos(self.vehicle_yaw)
        sin_yaw = sin(self.vehicle_yaw)
        cross_track_error = dx * sin_yaw - dy * cos_yaw

        if abs(cross_track_error) < 100:
            self.errors.append(abs(cross_track_error))

        cross_track_error = np.clip(cross_track_error, -self.max_cross_track_error, self.max_cross_track_error)

        path_point_local_x = cos(self.vehicle_yaw) * (closest_point.x - self.current_position.x) + sin(self.vehicle_yaw) * (closest_point.y - self.current_position.y)
        path_point_local_y = -sin(self.vehicle_yaw) * (closest_point.x - self.current_position.x) + cos(self.vehicle_yaw) * (closest_point.y - self.current_position.y)

        if path_point_local_y > 0:
            cross_track_error = abs(cross_track_error)
        else:
            cross_track_error = -abs(cross_track_error)

        dx = closest_point.x - self.current_position.x
        dy = closest_point.y - self.current_position.y
        path_heading = atan2(dy, dx)
        heading_error = path_heading - self.vehicle_yaw

        # [-pi, pi] 범위로 변환
        while heading_error > pi:
            heading_error -= 2 * pi
        while heading_error < -pi:
            heading_error += 2 * pi

        # alpha
        alpha = self.alpha / max(self.status_msg.velocity.x, 0.1)

        CTR = atan2(self.k * cross_track_error, self.target_velocity)
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

# if __name__ == "__main__":
#     rospy.init_node('path_tracking_node', anonymous=True)

#     pure_pursuit_controller = pure_pursuit_no_npc()
#     stanley_controller = stanley_parking()

#     rate = rospy.Rate(10)  # 10 Hz
#     while not rospy.is_shutdown():
#         if stanley_controller.is_path and stanley_controller.is_odom and stanley_controller.is_status:
#             stanley_controller.calc_stanley_control()
#         rate.sleep()

#     # End the simulation by setting the end time
#     stanley_controller.set_end_time()

#     if stanley_controller.global_path is not None:
#         mean_error, max_error, variance = stanley_controller.calculate_statistics()
#         plot_paths(stanley_controller.global_path, stanley_controller.x_ego, stanley_controller.y_ego,
#                    stanley_controller.calculate_total_time(), variance, mean_error, max_error)