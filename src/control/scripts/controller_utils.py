#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import matplotlib.pyplot as plt
import numpy as np
import time


def get_waypoint(ego_status, global_path):
    min_dist = float('inf')
    current_waypoint = -1
    for i, pose in enumerate(global_path.poses):
        dx = ego_status.position.x - pose.pose.position.x
        dy = ego_status.position.y - pose.pose.position.y
        dist = np.sqrt(dx**2 + dy**2)
        if min_dist > dist:
            min_dist = dist
            current_waypoint = i
    return current_waypoint


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


def calculate_statistics(errors):
    if len(errors) > 0:
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        variance = np.var(errors)
        return mean_error, max_error, variance
    return 0.0, 0.0, 0.0

def calculate_total_time(start_time, end_time=None):
    if start_time is not None and end_time is not None:
        return end_time - start_time
    elif start_time is not None:
        return time.time() - start_time
    return 0.0

def set_end_time(end_time):
    if end_time is None:
        return time.time()
    return end_time

def unified_calculator(errors=None, start_time=None, end_time=None, operation=None):
    if operation == 'statistics':
        return calculate_statistics(errors)
    elif operation == 'total_time':
        return calculate_total_time(start_time, end_time)
    elif operation == 'set_end_time':
        return set_end_time(end_time)
    else:
        raise ValueError("Invalid operation specified")
    