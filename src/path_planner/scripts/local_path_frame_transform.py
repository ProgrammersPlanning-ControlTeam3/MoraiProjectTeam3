#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import cos, sin, sqrt, pow, pi
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from morai_msgs.msg import EgoVehicleStatus

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
        self.yaw = msg.heading * (pi / 180.0)  # Convert degrees to radians

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

        # 현재 위치를 프레넷 좌표로 변환
        s, d = get_frenet(x, y, mapx, mapy)

        # 목표 위치 설정
        end_idx = min(start_idx + self.local_path_size, len(self.global_path_msg.poses))

        target_point = self.global_path_msg.poses[end_idx - 1].pose.position
        s_target, d_target = get_frenet(target_point.x, target_point.y, mapx, mapy)

        # Quintic Polynomial 계수 계산 (시간 T는 임의로 설정, 예: 1초)
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

def get_frenet(x, y, mapx, mapy):
    next_wp = next_waypoint(x, y, mapx, mapy)
    prev_wp = next_wp - 1

    n_x = mapx[next_wp] - mapx[prev_wp]
    n_y = mapy[next_wp] - mapy[prev_wp]
    x_x = x - mapx[prev_wp]
    x_y = y - mapy[prev_wp]

    proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)
    proj_x = proj_norm * n_x
    proj_y = proj_norm * n_y

    #-------- get frenet d
    frenet_d = get_dist(x_x, x_y, proj_x, proj_y)

    ego_vec = [x - mapx[prev_wp], y - mapy[prev_wp], 0]
    map_vec = [n_x, n_y, 0]
    d_cross = np.cross(ego_vec, map_vec)
    
    if d_cross[-1] > 0:
        frenet_d = -frenet_d

    #-------- get frenet s
    frenet_s = 0
    for i in range(prev_wp):
        frenet_s += get_dist(mapx[i], mapy[i], mapx[i + 1], mapy[i + 1])

    frenet_s += get_dist(0, 0, proj_x, proj_y)

    return frenet_s, frenet_d

def get_cartesian(s, d, mapx, mapy, maps):
    prev_wp = 0

    while (s > maps[prev_wp + 1]) and (prev_wp < len(maps) - 2):
        prev_wp += 1

    next_wp = np.mod(prev_wp + 1, len(mapx))

    dx = (mapx[next_wp] - mapx[prev_wp])
    dy = (mapy[next_wp] - mapy[prev_wp])  # 수정된 부분

    heading = np.arctan2(dy, dx)  # [rad]

    # the x, y, s along the segment
    seg_s = s - maps[prev_wp]

    seg_x = mapx[prev_wp] + seg_s * np.cos(heading)
    seg_y = mapy[prev_wp] + seg_s * np.sin(heading)  # 수정된 부분

    perp_heading = heading + 90 * np.pi / 180
    x = seg_x + d * np.cos(perp_heading)
    y = seg_y + d * np.sin(perp_heading)

    return x, y, heading


def next_waypoint(x, y, mapx, mapy):
    closest_wp = get_closest_waypoints(x, y, mapx, mapy)
    map_x = mapx[closest_wp]
    map_y = mapy[closest_wp]

    heading = np.arctan2((map_y - y), (map_x - x))
    angle = np.abs(np.arctan2(np.sin(heading), np.cos(heading)))

    if angle > np.pi / 4:
        closest_wp += 1
        if closest_wp == len(mapx):
            closest_wp = 0

    return closest_wp

def get_closest_waypoints(x, y, mapx, mapy):
    min_len = 1e10
    closest_wp = 0

    for i in range(len(mapx)):
        _mapx = mapx[i]
        _mapy = mapy[i]
        dist = get_dist(x, y, _mapx, _mapy)

        if dist < min_len:
            min_len = dist
            closest_wp = i

    return closest_wp

def get_dist(x, y, _x, _y):
    return np.sqrt((x - _x) ** 2 + (y - _y) ** 2)

if __name__ == '__main__':
    try:
        PathPub()
    except rospy.ROSInterruptException:
        pass
