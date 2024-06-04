#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import cos, sin, sqrt, pow, pi
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from morai_msgs.msg import EgoVehicleStatus
from frame_transform import *

class PathPub:
    def __init__(self):
        rospy.init_node('path_pub', anonymous=True)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/global_path", Path, self.global_path_callback)

        self.local_path_pub = rospy.Publisher('/local_path', Path, queue_size=1)

        self.global_path_msg = Path()
        self.global_path_msg.header.frame_id = 'map'

        self.is_status = False
        self.local_path_size = 65

        self.x = 0
        self.y = 0
        self.yaw = 0

        rate = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            if self.is_status and self.global_path_msg.poses:
                local_path_msg = Path()
                local_path_msg.header.frame_id = 'map'

                x = self.x
                y = self.y
                yaw = self.yaw

                current_waypoint = self.get_closest_waypoint(x, y)
                if current_waypoint != -1:
                    local_path_points = self.generate_local_path(x, y, yaw, current_waypoint)
                    for point in local_path_points:
                        tmp_pose = PoseStamped()
                        tmp_pose.pose.position.x = point[0]
                        tmp_pose.pose.position.y = point[1]
                        tmp_pose.pose.orientation.w = 1
                        local_path_msg.poses.append(tmp_pose)

                self.local_path_pub.publish(local_path_msg)
            rate.sleep()

    def status_callback(self, msg):
        self.is_status = True
        self.x = msg.position.x
        self.y = msg.position.y
        self.yaw = msg.heading * (pi / 180.0)

    def global_path_callback(self, msg):
        self.global_path_msg = msg

    def get_closest_waypoint(self, x, y):
        min_dis = float('inf')
        closest_waypoint = -1
        for i, waypoint in enumerate(self.global_path_msg.poses):
            distance = sqrt(pow(x - waypoint.pose.position.x, 2) + pow(y - waypoint.pose.position.y, 2))
            if distance < min_dis:
                min_dis = distance
                closest_waypoint = i
        return closest_waypoint

    def generate_local_path(self, x, y, yaw, start_idx):
        local_path_points = []

        mapx = [pose.pose.position.x for pose in self.global_path_msg.poses]
        mapy = [pose.pose.position.y for pose in self.global_path_msg.poses]
        maps = [0]
        for i in range(1, len(mapx)):
            maps.append(maps[-1] + get_dist(mapx[i - 1], mapy[i - 1], mapx[i], mapy[i]))

        s, d = get_frenet(x, y, mapx, mapy)

        end_idx = min(start_idx + self.local_path_size, len(self.global_path_msg.poses))

        target_point = self.global_path_msg.poses[end_idx - 1].pose.position
        s_target, d_target = get_frenet(target_point.x, target_point.y, mapx, mapy)

        T = 1.0
        s_coeff = self.quintic_polynomial_coeffs(s, 0, 0, s_target, 0, 0, T)
        d_coeff = self.quintic_polynomial_coeffs(d, 0, 0, d_target, 0, 0, T)

        for i in range(self.local_path_size):
            t = i * (T / self.local_path_size)
            s_val = self.quintic_polynomial_value(s_coeff, t)
            d_val = self.quintic_polynomial_value(d_coeff, t)
            point_x, point_y, _ = get_cartesian(s_val, d_val, mapx, mapy, maps)
            local_path_points.append((point_x, point_y))

        return local_path_points

    def quintic_polynomial_coeffs(self, xs, vxs, axs, xe, vxe, axe, T):
        A = np.array([
            [0, 0, 0, 0, 0, 1],
            [T**5, T**4, T**3, T**2, T, 1],
            [0, 0, 0, 0, 1, 0],
            [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [20*T**3, 12*T**2, 6*T, 2, 0, 0]
        ])
        B = np.array([xs, xe, vxs, vxe, axs, axe])
        X = np.linalg.solve(A, B)
        return X

    def quintic_polynomial_value(self, coeffs, t):
        return coeffs[0]*t**5 + coeffs[1]*t**4 + coeffs[2]*t**3 + coeffs[3]*t**2 + coeffs[4]*t + coeffs[5]



if __name__ == '__main__':
    try:
        PathPub()
    except rospy.ROSInterruptException:
        pass
