#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import pi
import os
import numpy as np
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)

sys.path.insert(0, '/home/ubuntu/MoraiProjectTeam3/src')

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from morai_msgs.msg import EgoVehicleStatus
from tf.transformations import euler_from_quaternion
from frame_transform import get_frenet, get_cartesian, get_dist
from parking.scripts.dubins import Dubins
from weightedLeastSquare import WeightedLeastSquare



"""
1. Edited global path(add yaw)
2.

"""
class PathPub:
    def __init__(self):
        rospy.init_node('path_pub', anonymous=True)
        # rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.ego_callback)

        #parking_path Sub
        rospy.Subscriber("/global_path_parking", Path, self.global_path_parking_callback)


        self.local_path_pub = rospy.Publisher('/local_path', Path, queue_size=1)

        self.global_path_msg = Path()
        self.global_path_msg.header.frame_id = 'map'

        #Parking Path
        self.global_path_parking_msg = Path()
        self.global_path_parking_msg.header.frame_id = 'map'

        #weighted
        self.WLS = WeightedLeastSquare()

        self.is_status = False
        self.local_path_size = 30

        self.x = 0
        self.y = 0
        self.yaw = 0
        self.parkingLot = False
        self.isArrived = False
        rate = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            if self.is_status and self.global_path_msg.poses:
                if self.isArrived:
                    local_path_msg = self.create_local_path_msg()
                    self.local_path_pub.publish(local_path_msg)
            rate.sleep()

    def ego_callback(self,msg) :
        yaw = msg.heading + 360
        yaw = (yaw%360) * (pi / 180)
        self.yaw = yaw
    
    def odom_callback(self, msg):
        self.is_status=True

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        if self.arrivedAtPoint(7.23,1066.8476) :
            self.isArrived = True
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
        yaw = self.yaw
        local_path_points = self.generate_local_path(x, y)
        local_path = []
        ## dubins
        # dubins = Dubins()
        # kappa_ = 1./2.0
        # cartesian_path, _,_ = dubins.plan([x,y,yaw],local_path_points[-1],kappa_)
        # path_x , path_y , path_yaw = cartesian_path
        # for i in range(len(path_x)) :
        #     local_path.append([path_x[i],path_y[i]])
        # for point in local_path:
        #     tmp_pose = PoseStamped()
        #     tmp_pose.pose.position.x = point[0]
        #     tmp_pose.pose.position.y = point[1]
        #     tmp_pose.pose.orientation.w = 1
        #     local_path_msg.poses.append(tmp_pose)
        #weighted
        self.coeff = self.WLS.fit_curve(local_path_points)
        # print(self.coeff)
        x_range = [point[0] for point in local_path_points]
        #print("x range", x_range)
        y_range = self.evaluate_polynomial(self.coeff, x_range)
        # print("y_range", y_range)
        for i in range(len(x_range)):
            local_path.append([x_range[i], y_range[i]])
        for point in local_path:
            tmp_pose = PoseStamped()
            tmp_pose.pose.position.x = point[0]
            tmp_pose.pose.orientation.w = 1
            local_path_msg.poses.append(tmp_pose)
        return local_path_msg

    def generate_local_path(self, x, y):
        local_path_points = []

        #글로벌 path 기반 경로 생성.
        mapx = [pose.pose.position.x for pose in self.global_path_msg.poses]
        mapy = [pose.pose.position.y for pose in self.global_path_msg.poses]
        map_yaw = [pose.pose.orientation.w for pose in self.global_path_msg.poses]

        maps = [0]
        cnt = -1
        min_dist = 1e9
        for i in range(len(mapx)):
            dist = get_dist(self.x,self.y, mapx[i],mapy[i])
            if min_dist > dist :
                min_dist = dist
                cnt = i
        for i in range(cnt, cnt+10) :
            try:
                local_path_points.append([mapx[i],mapy[i],map_yaw[i]])
            except IndexError :
                pass
        return local_path_points

    def evaluate_polynomial(self, coefficients, x):
        """ 주어진 9차 다항식의 계수와 x값을 이용하여 y값 계산 """
        return np.polyval(list((coefficients)), x)
        

    def get_dist(self, x1, y1, x2, y2):
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    def arrivedAtPoint (self,x, y, gap = 5.0) :
        x1, y1 = self.x , self.y
        if self.get_dist(x1,y1,x,y) < gap :
            return True
        else:
            return False




if __name__ == '__main__':
    try:
        PathPub()
    except rospy.ROSInterruptException:
        pass
