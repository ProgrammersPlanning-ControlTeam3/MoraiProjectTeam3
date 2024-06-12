#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import pi, atan2, cos, sin
import numpy as np
from scipy.optimize import minimize
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from morai_msgs.msg import EgoVehicleStatus
from tf.transformations import euler_from_quaternion
from frame_transform import get_frenet, get_cartesian, get_dist
import sympy as sp


class PathPub:
    def __init__(self):
        rospy.init_node('path_pub', anonymous=True)
        # rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        '''
            주차장 Global Path만 받게 하는 것도 괜찮을 거 같아요.
        '''
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/global_path", Path, self.global_path_callback)

        self.local_path_pub = rospy.Publisher('/parking_local_path', Path, queue_size=1)

        self.global_path_msg = Path()
        self.global_path_msg.header.frame_id = 'map'

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

        mapx = [pose.pose.position.x for pose in self.global_path_msg.poses]
        mapy = [pose.pose.position.y for pose in self.global_path_msg.poses]
        yaw = [pose.pose.orientation.w for pose in self.global_path_msg.poses]
        maps = [0]
        for i in range(1, len(mapx)):
            maps.append(maps[-1] + get_dist(mapx[i - 1], mapy[i - 1], mapx[i], mapy[i]))
        # 차량의 s, d 좌표
        s, d = get_frenet(x, y, mapx, mapy)

        # Local Path 정보 생성 s=x, d = y #가장 가까운 s 값 뽑아내기
        s_target = s + min(self.local_path_size, maps[-1] - s)
        s_targets =[]
        d_targets =[]
        yaw_targets = []
        for i in range(len(maps)):
            if maps[i] >= s_target:
                start_index = i
                break
        for i in range(start_index, min(start_index+10, len(maps))):
            s_point, d_point = get_frenet(mapx[i], mapy[i], mapx, mapy)
            yaw_frenet= self.yaw_to_frenet(yaw[i], mapx, mapy, i)
            s_targets.append(s_point)
            d_targets.append(d_point)
            yaw_targets.append(yaw_frenet)
        sd_pairs = list(zip(s_targets, d_targets, yaw_targets))

        # 조정된 d_target 계산
        # d_adjustment = 0  # 중앙에 맞추기 위해 조정할 값
        # d_target += d_adjustment

        T = 1.0
        path_function = WeightedLeastSquare()
        coefficients = path_function.fit_curve(sd_pairs)

        for s_val in np.linspace(s, s+self.local_path_size, num=self.local_path_size):
            d_val = self.evaluate_polynomial(coefficients, s_val)
            point_x, point_y, _ = get_cartesian(s_val, d_val, mapx, mapy, maps)
            local_path_points.append((point_x, point_y))
        return local_path_points

    def yaw_to_frenet(self, yaw, mapx, mapy, index):
        """ 글로벌 경로 상에서 주어진 인덱스의 yaw 값을 Frenet 경로의 방향으로 변환합니다. """
        dx = mapx[index + 1] - mapx[index] if index < len(mapx) - 1 else mapx[index] - mapx[index - 1]
        dy = mapy[index + 1] - mapy[index] if index < len(mapy) - 1 else mapy[index] - mapy[index - 1]
        path_angle = atan2(dy, dx)
        frenet_yaw = yaw - path_angle  # 경로 탄젠트 각도와 yaw 각도의 차이 계산
        return frenet_yaw

    def evaluate_polynomial(self, coefficients, x):
        """ 주어진 9차 다항식의 계수와 x값을 이용하여 y값 계산 """
        return sum(c * x ** i for i, c in enumerate(reversed(coefficients)))


class WeightedLeastSquare:
    def __init__(self, degree=9, start_end_weight_multiplier=5):
        self.degree = degree
        self.start_end_weight_multiplier = start_end_weight_multiplier
        self.x_sym = sp.symbols('x')
        self.b = sp.symbols(f'b_0:{degree + 1}')
        self.polynomial_func = sum(self.b[i] * self.x_sym**(degree - i) for i in range(degree + 1))
        self.polynomial_derivative = sp.diff(self.polynomial_func, self.x_sym)

    def evaluate_derivative(self, x_values):
        # For Jacobian... Fitting Function
        f_prime = sp.lambdify(self.x_sym, self.polynomial_derivative, 'numpy')
        return f_prime(x_values)

    def fit_curve(self, points):
        # points는 [x, y, yaw] 형식의 리스트
        x_data = np.array([p[0] for p in points])
        y_data = np.array([p[1] for p in points])
        yaw_data = np.array([p[2] for p in points])

        # 각 점의 방향을 고려한 가중치 계산
        predicted_tangent = np.tan(yaw_data)
        actual_tangent = self.evaluate_derivative(x_data)
        weights = np.exp(-np.abs(predicted_tangent - actual_tangent))  # 방향 차이에 따른 가중치
        weights[0] *= self.start_end_weight_multiplier
        weights[-1] *= self.start_end_weight_multiplier

        #  Weight Function
        W = np.diag(weights)

        # Jacobian Matrix
        J_func = sp.lambdify(self.x_sym, self.polynomial_derivative, 'numpy')
        J_matrix = np.vstack([J_func(x) for x in x_data])

        A = J_matrix.T @ W @ J_matrix
        B = J_matrix.T @ W @ y_data
        coefficients = np.linalg.solve(A, B)

        return coefficients

if __name__ == '__main__':
    try:
        PathPub()
    except rospy.ROSInterruptException:
        pass
