#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import sqrt, pow

class VelocityPlanningFollowVehicle:
    def __init__(self, car_max_speed, time_gap):

        self.current_velocity = 0.0

        self.car_max_speed = car_max_speed
        self.time_gap = time_gap

    def control_velocity(self, local_path, object_data):

        desired_velocities = []

        for path in local_path.poses:
            desired_velocity = self.car_max_speed
            for npc in object_data.npc_list:
                dis = sqrt(pow(path.pose.position.x - npc.position.x, 2) + pow(path.pose.position.y - npc.position.y, 2))
                if dis < self.time_gap * self.current_velocity:
                    desired_velocity = min(desired_velocity, npc.velocity.x - 2.0)
            desired_velocities.append(desired_velocity)

        return desired_velocities

