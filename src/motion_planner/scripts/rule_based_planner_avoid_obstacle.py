#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import rospy
from math import cos, sin, pi, sqrt, pow, atan2
import numpy as np
from tf.transformations import euler_from_quaternion

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus, ObjectStatusList, EventInfo, Lamps
from morai_msgs.srv import MoraiEventCmdSrv
from morai_msgs.msg import CtrlCmd
import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)

# from lib.mgeo.class_defs import *
sys.path.insert(0, '/home/ubuntu/MoraiProjectTeam3/src')
#print(sys.path)

from control.scripts.pid_controller import pidControl
from control.scripts.lateral_controller import stanley
# from control.scripts.lateral_controller import pure_pursuit
from control.scripts.longitudinal_controller import velocityPlanning
from object_detector.scripts.object_detector import object_detector
from control.scripts.longitudinal_follow_vehicle import FollowVehicle

class rule_based_planner:
    def __init__(self):

        # Node Initialize
        rospy.init_node('rule_based_planner', anonymous=True)

        # Get the GLOBAL PATH INFO
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/lattice_path", Path, self.path_callback)
        # GET OBJECT TOPIC
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self.status_callback)
        # rospy.Subscriber("/Object_topic", data_class=ObjectStatusList, self.object_callback)
        # PUBLISHING CONTROL TOPIC, Name ctrl_cmd_pub Get CtrlCmd Topic
        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd_0', CtrlCmd, queue_size=1)
        # 객체 정보 수신
        self.ctrl_cmd_msg = CtrlCmd() # 차량 제어 명령
        self.ctrl_cmd_msg.longlCmdType = 1 # 특정 제어 모드임. CtrlCmd 참조할 것

        self.is_path = False # 경로 정보 유무 확인
        self.is_odom = False # 오도메트리 정보 유무 확인. 현재 위치 송수신
        self.is_status = False # 차량 상태 정보

        self.is_global_path = False # 글로벌 경로가 성공적으
        self.is_go = False # 차량이 움직이기 시작했는지 나타냄
        self.time = 1 # 동작의 시간 조절에 사용되는 숫자 변수
        self.parking_time=1 # 주차시간을 추적하거나 조절하는데 사용

        self.current_position = Point()
        self.is_look_forward_point = False

        self.target_velocity = 40 # morive max_speed": 60, default : 40

        # 제어 시스템 및 알고리즘 초기화 부분
        self.pid = pidControl() # PID Control
        self.vel_planning = velocityPlanning(self.target_velocity / 3.6, 0.15) # Velocity Control
        self.stanley = stanley() 
        self.follow_vehicle = FollowVehicle()
        # self.pure_pursuit = pure_pursuit() # Pure Pursuit control
        # self.object_detector = object_detector() # Object Detection to avoid

        # 무한 루프: self.is_global_path가 True로 설정될때까지 계속 실행.
        while True:
            if self.is_global_path == True: # If you get the Global Path Information
                self.velocity_list = self.vel_planning.curvedBaseVelocity(self.global_path, 50) # Set velocity ,,
                # 속도 프로파일 계산 후 아래 루프 실행
                break
            else:
                pass

        rate = rospy.Rate(30)  # 30hz
        print("path_info", self.is_path, "odom_info", self.is_odom, "status info", self.is_status)
        while not rospy.is_shutdown():
            if self.is_path == True and self.is_odom == True and self.is_status == True: # Everything is OK
                prev_time = time.time()

                self.current_waypoint = self.stanley.get_current_waypoint(self.status_msg, self.global_path)
                # self.current_waypoint = self.pure_pursuit.get_current_waypoint(self.status_msg, self.global_path)

                self.target_velocity = self.velocity_list[self.current_waypoint] * 3.6

                ## TODO target_velocity -> 감속 (앞 차량이 있거나, 예측 경로와 겹칠 경우)
                self.re_target_velocity = self.follow_vehicle.control_velocity(self.target_velocity)

                # steering = self.stanley.calc_stanley_control()
                # steering = self.pure_pursuit.calc_pure_pursuit()
                # TODO tollgate area : No lattice path. Follow local path. Need to make follow local path method in stanley class
                # steering = self.stanley.calc_stanley_control_local()

                if (self.status_msg.position.y < 1300):
                    steering = self.stanley.calc_stanley_control_local()
                else:
                    steering = self.stanley.calc_stanley_control()

                self.ctrl_cmd_msg.steering = steering #0.0 last

                output = self.pid.pid(self.re_target_velocity, self.status_msg.velocity.x * 3.6)

                if output > 0.0:
                    self.ctrl_cmd_msg.accel = output
                    self.ctrl_cmd_msg.brake = 0.0
                else:
                    self.ctrl_cmd_msg.accel = 0.0
                    self.ctrl_cmd_msg.brake = -output


                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

            rate.sleep()


    def call_service(self, gear_value):
        rospy.wait_for_service('Service_MoraiEventCmd')
        try:
            service_client = rospy.ServiceProxy('Service_MoraiEventCmd', MoraiEventCmdSrv)
            request_data = EventInfo()

            request_data.option = 2
            request_data.ctrl_mode = 4
            request_data.gear = gear_value
            lamps_data = Lamps()
            lamps_data.turnSignal = 0
            lamps_data.emergencySignal = 0
            request_data.lamps = lamps_data
            request_data.set_pause = False

            response = service_client(request_data)
            return response

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
            return None


    def object_callback(self, msg):
        self.is_object = True
        self.object_msg = msg


    def global_path_callback(self, msg):
        self.global_path = msg
        self.is_global_path = True

    def status_callback(self, msg):
        self.is_status=True
        self.status_msg=msg

    def odom_callback(self,msg):
        self.is_odom=True
        odom_quaternion=(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        _,_,self.vehicle_yaw=euler_from_quaternion(odom_quaternion)
        self.current_position.x=msg.pose.pose.position.x
        self.current_position.y=msg.pose.pose.position.y

    def path_callback(self,msg):
        self.is_path=True
        self.path=msg
    




if __name__ == '__main__':
    try:
        test_track = rule_based_planner()

    except rospy.ROSInterruptException:
        pass
