#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import pi, atan2
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from frame_transform import get_frenet, get_cartesian, get_dist

isArrived = False

class PathPub:
    def __init__(self):
        rospy.init_node('path_pub', anonymous=True)
        # rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/global_path", Path, self.global_path_callback)

        self.local_path_pub = rospy.Publisher('/local_path', Path, queue_size=1)

        self.global_path_msg = Path()
        self.global_path_msg.header.frame_id = 'map'

        #Parking Path
        self.global_path_parking_msg = Path()
        self.global_path_parking_msg.header.frame_id = 'map'


        self.is_status = False
        self.local_path_size = 30 # 30

        self.x = 0
        self.y = 0
        self.yaw = 0

        rate = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            if self.is_status and self.global_path_msg.poses:
                local_path_msg = self.create_local_path_msg()
                self.local_path_pub.publish(local_path_msg)
            rate.sleep()

    def odom_callback(self, msg):
        self.is_status = True

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        # print(self.x, self.y, self.yaw)

    def global_path_callback(self, msg):
        self.global_path_msg = msg

    def global_path_parking_callback(self, msg):
        self.global_path_parking_msg = msg

    def create_local_path_msg(self):
        local_path_msg = Path()
        local_path_msg.header.frame_id = 'map'

        x = self.x
        y = self.y
        yaw = self.yaw

        local_path_points = self.generate_local_path(x, y, yaw)
        for point in local_path_points:
            tmp_pose = PoseStamped()
            tmp_pose.pose.position.x = point[0]
            tmp_pose.pose.position.y = point[1]
            quaternion = quaternion_from_euler(0, 0, point[2])
            tmp_pose.pose.orientation.x = quaternion[0]
            tmp_pose.pose.orientation.y = quaternion[1]
            tmp_pose.pose.orientation.z = quaternion[2]
            tmp_pose.pose.orientation.w = quaternion[3]
            local_path_msg.poses.append(tmp_pose)

        return local_path_msg

    def generate_local_path(self, x, y, yaw):
        local_path_points = []

        #글로벌 path 기반 경로 생성.
        mapx = [pose.pose.position.x for pose in self.global_path_msg.poses]
        mapy = [pose.pose.position.y for pose in self.global_path_msg.poses]
        map_yaw = self.calculate_yaw_from_path(mapx, mapy)

        maps = [0]

        for i in range(1, len(mapx)):
            maps.append(maps[-1] + get_dist(mapx[i - 1], mapy[i - 1], mapx[i], mapy[i]))

        s, d = get_frenet(x, y, mapx, mapy)

        s_target = s + min(self.local_path_size, maps[-1] - s)

        d_target = None
        yaw_target = None
        for i in range(1, len(maps)):
            if maps[i] >= s_target:
                _, d_target = get_frenet(mapx[i], mapy[i], mapx, mapy)
                yaw_target = map_yaw[i]
                break

        if d_target is None:
            d_target = d
        if yaw_target is None:
            yaw_target = yaw

        # # 조정된 d_target 계산
        # d_adjustment = 0.7  # 중앙에 맞추기 위해 조정할 값
        # d_target += d_adjustment

        # 5차 곡선 생성
        T = 1.0
        s_coeff = self.generate_5th_order_polynomial(s, s_target, 0, T)
        d_coeff = self.generate_5th_order_polynomial(d, d_target, 0, T)
        yaw_coeff = self.generate_5th_order_polynomial(yaw, yaw_target, 0, T)

        for i in range(self.local_path_size):
            t = i * (T / self.local_path_size)
            s_val = self.calculate_polynomial(s_coeff, t)
            d_val = self.calculate_polynomial(d_coeff, t)
            yaw_val = self.calculate_polynomial(yaw_coeff, t)

            if s_val > maps[-1]:
                s_val = maps[-1]

            point_x, point_y, _ = get_cartesian(s_val, d_val, mapx, mapy, maps)
            # print(point_x, point_y, yaw_val)
            local_path_points.append((point_x, point_y, yaw_val))

        return local_path_points

    def calculate_yaw_from_path(self, mapx, mapy):
        map_yaw = []
        for i in range(len(mapx) - 1):
            dx = mapx[i + 1] - mapx[i]
            dy = mapy[i + 1] - mapy[i]
            yaw_angle = atan2(dy, dx)
            map_yaw.append(yaw_angle)
        map_yaw.append(map_yaw[-1])
        return map_yaw

    def generate_5th_order_polynomial(self, ys, yf, xs, xf):
        # 5차 곡선 계수 계산
        if xf == 0:
            return [ys, 0, 0, 0, 0, 0]  # xf가 0일 경우를 처리
        a0 = ys
        a1 = 0
        a2 = 0
        a3 = (10 * (yf - ys)) / (xf ** 3)
        a4 = (-15 * (yf - ys)) / (xf ** 4)
        a5 = (6 * (yf - ys)) / (xf ** 5)
        return [a0, a1, a2, a3, a4, a5]

    def calculate_polynomial(self, a, x_vals):
        return a[0] + a[1] * x_vals + a[2] * x_vals**2 + a[3] * x_vals**3 + a[4] * x_vals**4 + a[5] * x_vals**5

if __name__ == '__main__':
    try:
        PathPub()
    except rospy.ROSInterruptException:
        pass
