#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from morai_msgs.msg import EgoVehicleStatus
import matplotlib.pyplot as plt
import cvxpy as cp
import time
from controller_utils import *


class MPCController:
    def __init__(self):
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/lattice_path", Path, self.path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)

        self.is_path = False
        self.is_odom = False
        self.is_status = False
        self.is_global_path = False

        self.current_position = Point()
        self.vehicle_length = 5.205  # Hyundai Ioniq (hev)
        self.horizon = 10  # MPC horizon
        self.dt = 0.01  # Time step

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

    def status_callback(self, msg):
        self.is_status = True
        self.status_msg = msg

    def global_to_local(self, global_path, current_position, current_yaw):
        local_path = []
        for pose in global_path.poses:
            dx = pose.pose.position.x - current_position.x
            dy = pose.pose.position.y - current_position.y

            local_x = dx * math.cos(-current_yaw) - dy * math.sin(-current_yaw)
            local_y = dx * math.sin(-current_yaw) + dy * math.cos(-current_yaw)

            local_path.append((local_x, local_y))
        return local_path


    def mpc_control(self):
        if not self.is_path or not self.is_status:
            return 0.0, 0.0

        N = self.horizon  # MPC horizon
        dt = self.dt  # Time step

        # Convert path to local frame (global frame to local frame)
        local_path = self.global_to_local(self.path, self.current_position, self.vehicle_yaw)

        # Initialize optimization variables
        x = cp.Variable(N+1)
        y = cp.Variable(N+1)
        theta = cp.Variable(N+1)
        delta = cp.Variable(N)

        # Initialize cost and constraints
        cost = 0
        constraints = []

        # Initial conditions
        constraints += [x[0] == 0]
        constraints += [y[0] == 0]
        constraints += [theta[0] == 0]

        for t in range(N):
            ref_path = local_path[min(t, len(local_path)-1)]

            # Add cost terms for position and steering
            cost += 0.1 * cp.square(x[t+1] - ref_path[0])   # Minimize x position error
            cost += 5.0 * cp.square(y[t+1] - ref_path[1])   # Minimize y position error
            cost += 0.2 * cp.square(delta[t])               # Minimize steering angle

            # Smooth steering changes
            if t < N - 1:
                cost += 200.0 * cp.square(delta[t+1] - delta[t])
                constraints += [cp.abs(delta[t+1] - delta[t]) <= np.pi / 16]

            # Kinematic model constraints
            constraints += [x[t+1] == x[t] + self.status_msg.velocity.x * dt]               # Update x position
            constraints += [y[t+1] == y[t] + self.status_msg.velocity.x * theta[t] * dt]    # Update y position
            """       
                현재
                1) x_t+1 = x_t + v*dt
                2) y_t+1 = y_t + v*theta*dt
                
                원래는 아래처럼 하고싶었는데, cp.cos, cp.sin, np.sin, np.cos 모두 사용 불가능 하더라구요..
                각도 변화가 작다고 가정하고 위에 처럼 해놨습니다.
                1) x_t+1 = x_t + v*cos(theta)*dt
                2) y_t+1 = y_t + v*sin(theta)*dt
                
            """            
            constraints += [theta[t+1] == theta[t] + (self.status_msg.velocity.x / self.vehicle_length) * delta[t] * dt]    # Update heading angle
            constraints += [cp.abs(delta[t]) <= np.pi / 4]        # limits Steering angle 

        # Solve optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        # Return steering angle
        if delta.value is not None and len(delta.value) > 0:
            steering_angle = delta.value[0]
            self.calculate_error()
            return steering_angle
        else:
            return 0.0


    def calculate_error(self):
        if self.global_path and len(self.x_ego) > 0 and len(self.y_ego) > 0:
            ego_position = np.array([self.x_ego[-1], self.y_ego[-1]])
            min_dist = float('inf')
            for pose in self.global_path.poses:
                global_point = np.array([pose.pose.position.x, pose.pose.position.y])
                dist = np.linalg.norm(ego_position - global_point)
                if dist < min_dist:
                    min_dist = dist
            self.errors.append(min_dist)


    def get_current_waypoint(self, ego_status, global_path):
        min_dist = float('inf')
        current_waypoint = -1
        for i, pose in enumerate(global_path.poses):
            dx = ego_status.position.x - pose.pose.position.x
            dy = ego_status.position.y - pose.pose.position.y

            dist = np.sqrt(pow(dx, 2) + pow(dy, 2))
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


if __name__ == "__main__":
    rospy.init_node('mpc_path_tracking_node', anonymous=True)

    mpc_controller = MPCController()

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if mpc_controller.is_path and mpc_controller.is_odom and mpc_controller.is_status:
            steering = mpc_controller.mpc_control()
            # Control commands should be published to the vehicle here
        rate.sleep()

    mpc_controller.set_end_time()

    if mpc_controller.global_path is not None:
        mean_error, max_error, variance = mpc_controller.calculate_statistics()
        plot_paths(mpc_controller.global_path, mpc_controller.x_ego, mpc_controller.y_ego,
                   mpc_controller.calculate_total_time(), variance, mean_error, max_error)
