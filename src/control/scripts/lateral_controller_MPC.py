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

        N = self.horizon  # Prediction horizon
        dt = self.dt  # Time step

        # Local path conversion
        local_path = self.global_to_local(self.path, self.current_position, self.vehicle_yaw)

        # Initialize optimization variables
        x = cp.Variable(N+1)
        y = cp.Variable(N+1)
        theta = cp.Variable(N+1)
        delta = cp.Variable(N)
        # a = cp.Variable(N)

        # Initialize cost and constraints
        cost = 0
        constraints = []

        # Initial conditions
        constraints += [x[0] == 0]  # Local coordinates: initial x = 0
        constraints += [y[0] == 0]  # Local coordinates: initial y = 0
        constraints += [theta[0] == 0]  # Local coordinates: initial heading angle = 0

        for t in range(N):
            # Get the reference path point in local coordinates
            ref_path = local_path[min(t, len(local_path)-1)]

            # Define the cost function as the deviation from the reference path
            cost += 0.1 * cp.square(x[t+1] - ref_path[0])  # x error cost
            cost += 1.0 * cp.square(y[t+1] - ref_path[1])  # y error cost

            # Add a regularization term to minimize the control input
            cost += 0.1 * cp.square(delta[t])
            # cost += 0.1 * cp.square(a[t])

            if t < N - 1:
                # Penalize changes in steering to smooth the trajectory
                cost += 100.0 * cp.square(delta[t+1] - delta[t])
                # cost += 10.0 * cp.square(a[t+1] - a[t])

            # Vehicle kinematics constraints using linear approximation
            constraints += [x[t+1] == x[t] + self.status_msg.velocity.x * dt]
            constraints += [y[t+1] == y[t] + self.status_msg.velocity.x * theta[t] * dt]
            constraints += [theta[t+1] == theta[t] + (self.status_msg.velocity.x / self.vehicle_length) * delta[t] * dt]

            # Steering angle and acceleration limits
            constraints += [cp.abs(delta[t]) <= np.pi / 8]  # Reduce steering angle limit to π/8
            # constraints += [cp.abs(a[t]) <= 1.0]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        # Print y[t+1] values after solving the problem
        print("Optimized y values:")
        for t in range(N+1):
            print(f'y[{t}] = {y[t].value}')

        print("Optimized delta values:")
        for t in range(N):
            print(f'delta[{t}] = {delta[t].value}')

        if delta.value is not None and len(delta.value) > 0:
            steering_angle = delta.value[0]
            return steering_angle
        else:
            return 0.0






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

    plt.text(0.95, 0.05, f'Total Time: {total_time:.2f}s\nVariance: {variance:.4f}\nMean Error: {mean_error:.4f}\nMax Error: {max_error:.4f}', 
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=plt.gca().transAxes)

    plt.show()

# if __name__ == "__main__":
#     rospy.init_node('mpc_path_tracking_node', anonymous=True)

#     mpc_controller = MPCController()

#     rate = rospy.Rate(10)  # 10 Hz
#     while not rospy.is_shutdown():
#         if mpc_controller.is_path and mpc_controller.is_odom and mpc_controller.is_status:
#             steering = mpc_controller.mpc_control()
#             # Control commands should be published to the vehicle here
#         rate.sleep()

#     mpc_controller.set_end_time()

#     if mpc_controller.global_path is not None:
#         mean_error, max_error, variance = mpc_controller.calculate_statistics()
#         plot_paths(mpc_controller.global_path, mpc_controller.x_ego, mpc_controller.y_ego,
#                    mpc_controller.calculate_total_time(), variance, mean_error, max_error)
