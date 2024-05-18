#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import rospy
from math import cos, sin, pi, sqrt, atan2
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList
from geometry_msgs.msg import Point, PoseStamped, Point32
from nav_msgs.msg import Path
import numpy as np

class ManeuverBasedLatticePlanner:
### IMPLEMENTATION CODE AS FEATURE
  def __init__(self):
    #(1) Subscriber, Publisher
    rospy.Subscriber("/local_path", Path, self.path_callback)
    rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
    rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_callback)

    self.maneuver_lattice_path_pub=rospy.Publisher('/maneuver_lattice_path', Path, queue_size=1)

    self.is_path=False
    self.is_status=False
    self.is_obj=False
    rate=rospy.Rate(30)
    while not rospy.is_shutdown():
      if self.is_path and self.is_status and self.is_obj:
        if self.checkObject(self.local_path, self.object_data):
          maneuver_lattice_path=self.ManeuverBasedLatticePlanner()
          maneuver_lattice_path_index=self.collision_check(self.object_data, maneuver_lattice_path)

  def checkObject(self, ref_path, obejct_data):
        is_crash = False
        for npc in object_data.npc_list:
            for path in ref_path.poses:  
                dis = sqrt(pow(path.pose.position.x - npc.position.x, 2) + pow(path.pose.position.y - npc.position.y, 2))                
                # print(dis)
                if dis < 2.35: # 장애물의 좌표값이 지역 경로 상의 좌표값과의 직선거리가 2.35 미만일때 충돌이라 판단.
                    is_crash = True
                    print("crash!")
                    break

        return is_crash
  def collision_check(self, object_data, out_path):

        selected_lane = -1
        lane_weight = [3, 2, 1, 1, 2, 3] #reference path

        for obstacle in object_data.npc_list:
            for path_num in range(len(out_path)) :
                for path_pos in out_path[path_num].poses :
                    dis = sqrt(pow(obstacle.position.x - path_pos.pose.position.x, 2) + pow(obstacle.position.y - path_pos.pose.position.y, 2))
                    # if dis < 1.5: # 1.5
                        # lane_weight[path_num] = lane_weight[path_num] + 100

                    # weight based on distance
                    if dis < 10:
                        weight_increase = 200
                    elif dis < 25:
                        weight_increase = 100
                    elif dis < 40:
                        weight_increase = 50
                    else:
                        weight_increase = 0
                    lane_weight[path_num] += weight_increase

        selected_lane = lane_weight.index(min(lane_weight))     

        return selected_lane

  def path_callback(self,msg):
        self.is_path = True
        self.local_path = msg

  def status_callback(self,msg): ## Vehicl Status Subscriber
        self.is_status = True
        self.status_msg = msg

  def object_callback(self,msg):
        self.is_obj = True
        self.object_data = msg

  def ManeuverBasedLatticePlanner(self,ref_path, vehicle_status):
        out_path = []
        vehicle_pose_x = vehicle_status.position.x
        vehicle_pose_y = vehicle_status.position.y
        vehicle_velocity = vehicle_status.velocity.x * 3.6

        look_distance = int(vehicle_velocity * 0.2 * 2)

#TODO(1): Get the Vehicle Information
#TODO(2): Determine Maneuver Lane change or Lane Keeping
#TODO(3): Create Local Path to overtake leading vehicle(LC)
#TODO(4): Control The velocity to keeping distance to leading vehicles(LK)