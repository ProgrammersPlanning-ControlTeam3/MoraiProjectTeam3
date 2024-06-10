#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import time
import rospy
import rospkg
from math import cos,sin,pi,sqrt,pow,atan2
from geometry_msgs.msg import Point,PoseWithCovarianceStamped
from nav_msgs.msg import Odometry,Path
from morai_msgs.msg import CtrlCmd,EgoVehicleStatus
import numpy as np
import tf
from tf.transformations import euler_from_quaternion,quaternion_from_euler

sys.path.insert(0, '/home/ubuntu/final_project/src')
from control.scripts.pid_controller import pidControl
from control.scripts.longitudinal_controller import velocityPlanning

# advanced_purepursuit 은 차량의 차량의 종 횡 방향 제어 예제입니다.
# Purpusuit 알고리즘의 Look Ahead Distance 값을 속도에 비례하여 가변 값으로 만들어 횡 방향 주행 성능을 올립니다.
# 횡방향 제어 입력은 주행할 Local Path (지역경로) 와 차량의 상태 정보 Odometry 를 받아 차량을 제어 합니다.
# 종방향 제어 입력은 목표 속도를 지정 한뒤 목표 속도에 도달하기 위한 Throttle control 을 합니다.
# 종방향 제어 입력은 longlCmdType 1(Throttle control) 이용합니다.

# 노드 실행 순서 
# 1. subscriber, publisher 선언
# 2. 속도 비례 Look Ahead Distance 값 설정
# 3. 좌표 변환 행렬 생성
# 4. Steering 각도 계산
# 5. PID 제어 생성
# 6. 도로의 곡률 계산
# 7. 곡률 기반 속도 계획
# 8. 제어입력 메세지 Publish

class pure_pursuit :
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=True)

        #TODO: (1) subscriber, publisher 선언h_callback)
        rospy.Subscriber("/global_path", Path, self.global_path_callback)  
        rospy.Subscriber("/local_path", Path, self.path_callback)
        # rospy.Subscriber("/lattice_path", Path, self.path_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic",EgoVehicleStatus, self.status_callback)
        
        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd_0',CtrlCmd, queue_size=1)

        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 1

        ## edited for debug
        self.is_path = False
        self.is_odom = False
        self.is_status = False
        self.is_global_path = False
        self.is_look_forward_point = False
        
        self.forward_point = Point()
        self.current_position = Point()

        self.vehicle_length = 4.470 # wheel base 2.7 Length(m) : 4.470

        # self.lfd = 9.5
        # self.min_lfd = 5
        # self.max_lfd = 200
        # self.lfd_gain = 3.0
        # self.target_velocity = 40.0   # default = 40

        # self.lfd = 5
        # self.min_lfd = 1
        # self.max_lfd = 80
        # self.lfd_gain = 1.3

        # Test Jeonghoon
        self.lfd = 0.0
        self.min_lfd = 2.7 # 1 # 0.8
        self.max_lfd = 30 #  80 # 60
        self.lfd_gain = 1.0 # 1.3
        self.target_velocity = 35

        #########
        self.pid = pidControl()
        self.vel_planning = velocityPlanning(self.target_velocity/3.6, 0.15)

        while True:
            if self.is_global_path == True:
                self.velocity_list = self.vel_planning.curvedBaseVelocity(self.global_path, 70) # point_num 50 -> 35
                # rospy.loginfo("Velocity List: %s", self.velocity_list[:25])
                rospy.loginfo('Received global path data')
                break
            else:
                pass
                # rospy.loginfo('Waiting global path data')

        rate = rospy.Rate(30) # 30 -> 50 -> 30 Hz

        while not rospy.is_shutdown():

            ## can not enter to if
            if self.is_path == True and self.is_odom == True and self.is_status == True:
                prev_time = time.time()
                ## Added for debug
                rospy.loginfo("entered if")
                self.current_waypoint = self.get_current_waypoint(self.status_msg,self.global_path)
                # rospy.loginfo("current_waypoint: %s", self.current_waypoint)
                self.target_velocity = self.velocity_list[self.current_waypoint]*3.6
                # rospy.loginfo("target_velocity: %s", self.target_velocity)
                steering = self.calc_pure_pursuit()
                # rospy.loginfo("steering: %s", steering)

                if self.is_look_forward_point :
                    self.ctrl_cmd_msg.steering = steering
                else :
                    rospy.loginfo("no found forward point")
                    self.ctrl_cmd_msg.steering = 0.0

                output = self.pid.pid(self.target_velocity,self.status_msg.velocity.x*3.6)
                # rospy.loginfo("output: %s", output)

                # if output > -17.0: #default 0.0
                #     self.ctrl_cmd_msg.accel = output
                #     self.ctrl_cmd_msg.brake = 0.0
                # else:
                #     self.ctrl_cmd_msg.accel = 0.0
                #     self.ctrl_cmd_msg.brake = -output
                #     print (output)
                if output > 0.0:
                    self.ctrl_cmd_msg.accel = output # output  0 ~ 1.0
                    self.ctrl_cmd_msg.brake = 0.0
                    # morive brake tunning
                elif -8.0 < output <= 0.0:
                    self.ctrl_cmd_msg.accel = 0.0
                    self.ctrl_cmd_msg.brake = 0.0
                else:
                    self.ctrl_cmd_msg.accel = 0.0
                    self.ctrl_cmd_msg.brake = 0.35 # -output # output 0 ~ 1.0

                ## boost
                # if(self.status_msg.position.x > -29.0 and -105.0 < self.status_msg.position.y < -100.0):
                #     self.ctrl_cmd_msg.steering = 0.0
                #     self.ctrl_cmd_msg.accel = 0.3
                #     self.ctrl_cmd_msg.brake = 0.0
                #     self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

                ## Stop at end point
                if(self.status_msg.position.x > 157.0):
                    self.ctrl_cmd_msg.steering = 0.0
                    self.ctrl_cmd_msg.accel = 0.0
                    self.ctrl_cmd_msg.brake = 1.0
                    self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
                    print("Finished.")
                    # print("distance from barrels :", 171.34324645996094-self.status_msg.position.x-self.vehicle_length)

                #TODO: (8) 제어입력 메세지 Publish
                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

            rate.sleep()

    def path_callback(self,msg):
        self.is_path=True
        self.path=msg

    def status_callback(self,msg): ## Vehicle Status Subscriber
        self.is_status=True
        self.status_msg=msg

    def odom_callback(self,msg):
        self.is_odom=True
        odom_quaternion=(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        _,_,self.vehicle_yaw=euler_from_quaternion(odom_quaternion)
        self.current_position.x=msg.pose.pose.position.x
        self.current_position.y=msg.pose.pose.position.y

    def global_path_callback(self,msg):
        self.global_path = msg
        self.is_global_path = True

    def get_current_waypoint(self,ego_status,global_path):
        min_dist = float('inf')
        current_waypoint = -1
        for i,pose in enumerate(global_path.poses):
            dx = ego_status.position.x - pose.pose.position.x
            dy = ego_status.position.y - pose.pose.position.y

            dist = sqrt(pow(dx,2)+pow(dy,2))
            if min_dist > dist :
                min_dist = dist
                current_waypoint = i
        return current_waypoint

    def calc_pure_pursuit(self,):

        #TODO: (2) 속도 비례 Look Ahead Distance 값 설정
        self.lfd = (self.status_msg.velocity.x) * self.lfd_gain

        if self.lfd < self.min_lfd : 
            self.lfd=self.min_lfd
        elif self.lfd > self.max_lfd :
            self.lfd=self.max_lfd

        # rospy.loginfo("lfd: %s", self.lfd)

        vehicle_position=self.current_position
        self.is_look_forward_point= False

        translation = [vehicle_position.x, vehicle_position.y]

        #TODO: (3) 좌표 변환 행렬 생성
        trans_matrix = np.array([
                [cos(self.vehicle_yaw), -sin(self.vehicle_yaw),translation[0]],
                [sin(self.vehicle_yaw),cos(self.vehicle_yaw),translation[1]],
                [0                    ,0                    ,1            ]])

        det_trans_matrix = np.linalg.inv(trans_matrix)

        # for num,i in enumerate(self.path.poses):
        #     path_point=i.pose.position

        #     global_path_point = [path_point.x,path_point.y,1]
        #     local_path_point = det_trans_matrix.dot(global_path_point)    

        #     if local_path_point[0] > 0 :
        #         dis = sqrt(pow(local_path_point[0],2)+pow(local_path_point[1],2))
        #         if dis >= self.lfd :x
        #             self.forward_point = path_point
        #             self.is_look_forward_point = True
        #             break

        steering = 0.0  # init sterring

        for num, i in enumerate(self.path.poses):
            path_point = i.pose.position

            global_path_point = [path_point.x, path_point.y, 1]
            local_path_point = det_trans_matrix.dot(global_path_point)

            if local_path_point[0] > 0 :
                dis = sqrt(pow(local_path_point[0], 2) + pow(local_path_point[1], 2))
                if dis >= self.lfd:
                    self.forward_point = path_point
                    self.is_look_forward_point = True
                    #TODO: (4) Steering 각도 계산
                    theta = atan2(local_path_point[1], local_path_point[0])
                    steering = atan2((2 * self.vehicle_length * sin(theta)), self.lfd)
                    break

        # #TODO: (4) Steering 각도 계산
        # theta = atan2(local_path_point[1],local_path_point[0])
        # # rospy.loginfo("local_path_point: %s", local_path_point)
        # # rospy.loginfo("theta: %s", theta)
        # steering = atan2((2*self.vehicle_length*sin(theta)),self.lfd)

        return steering


if __name__ == '__main__':
    try:
        test_track=pure_pursuit()
    except rospy.ROSInterruptException:
        pass