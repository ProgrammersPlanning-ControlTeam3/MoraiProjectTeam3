#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import pi
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from morai_msgs.msg import EgoVehicleStatus
from tf.transformations import euler_from_quaternion
from frame_transform import get_frenet, get_cartesian, get_dist

isArrived = False

class PathPub:
    def __init__(self):
        rospy.init_node('path_pub', anonymous=True)
        # rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/global_path", Path, self.global_path_callback)

        #parking_path Sub
        rospy.Subscriber("/global_path_parking", Path, self.global_path_parking_callback)


        self.local_path_pub = rospy.Publisher('/local_path', Path, queue_size=1)

        self.global_path_msg = Path()
        self.global_path_msg.header.frame_id = 'map'

        #Parking Path
        self.global_path_parking_msg = Path()
        self.global_path_parking_msg.header.frame_id = 'map'


        self.is_status = False
        self.local_path_size = 30

        self.x = 0
        self.y = 0

        rate = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            if self.is_status and self.global_path_msg.poses:
                local_path_msg = self.create_local_path_msg()
                self.local_path_pub.publish(local_path_msg)
            rate.sleep()

    def odom_callback(self, msg):
        self.is_status=True

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

    # def status_callback(self, msg):
    #     self.is_status = True
    #     self.x = msg.position.x
    #     self.y = msg.position.y
    #     self.yaw = msg.heading * (pi / 180.0)

    def global_path_callback(self, msg):
        self.global_path_msg = msg

    def global_path_parking_callback(self, msg):
        self.global_path_parking_msg = msg

    def create_local_path_msg(self):
        local_path_msg = Path()
        local_path_msg.header.frame_id = 'map'

        x = self.x
        y = self.y

        local_path_points = self.generate_local_path(x, y)
        for point in local_path_points:
            tmp_pose = PoseStamped()
            tmp_pose.pose.position.x = point[0]
            tmp_pose.pose.position.y = point[1]
            tmp_pose.pose.orientation.w = 1
            local_path_msg.poses.append(tmp_pose)

        return local_path_msg

    def generate_local_path(self, x, y):
        local_path_points = []

        #글로벌 path 기반 경로 생성.
        mapx = [pose.pose.position.x for pose in self.global_path_msg.poses]
        mapy = [pose.pose.position.y for pose in self.global_path_msg.poses]
        maps = [0]
        
        for i in range(1, len(mapx)):
            maps.append(maps[-1] + get_dist(mapx[i - 1], mapy[i - 1], mapx[i], mapy[i]))

        s, d = get_frenet(x, y, mapx, mapy)

        s_target = s + min(self.local_path_size, maps[-1] - s)

        d_target = None
        for i in range(1, len(maps)):
            if maps[i] >= s_target:
                _, d_target = get_frenet(mapx[i], mapy[i], mapx, mapy)
                break

        if d_target is None:
            d_target = d

        # 조정된 d_target 계산
        d_adjustment = 0.7  # 중앙에 맞추기 위해 조정할 값
        d_target += d_adjustment

        T = 1.0
        s_coeff = self.quintic_polynomial_coeffs(s, 0, 0, s_target, 0, 0, T)
        d_coeff = self.quintic_polynomial_coeffs(d, 0, 0, d_target, 0, 0, T)

        for i in range(self.local_path_size):
            t = i * (T / self.local_path_size)
            s_val = self.quintic_polynomial_value(s_coeff, t)
            d_val = self.quintic_polynomial_value(d_coeff, t)

            if s_val > maps[-1]:
                s_val = maps[-1]

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