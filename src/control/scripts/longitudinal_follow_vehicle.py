#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from math import cos, sin, pi, sqrt, pow, atan2

from math import sqrt, pow
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList
from prediction.msg import TrackedPoint, PredictedObjectPath, PredictedObjectPathList, TrackedObjectPose, TrackedObjectPoseList
from geometry_msgs.msg import Point, PoseStamped, Point32
from nav_msgs.msg import Path
from std_msgs.msg import Int32

class FollowVehicle:
    def __init__(self):

        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_callback)
        rospy.Subscriber('/Object_topic/tracked_object_pose_topic', TrackedObjectPoseList, self.object_info_callback)
        rospy.Subscriber('/Object_topic/tracked_object_path_topic', PredictedObjectPathList, self.object_path_callback)
        rospy.Subscriber('/Object_topic/deleted_object_id', Int32, self.deleted_object_callback)
        rospy.Subscriber("/lattice_path", Path, self.lattice_path_callback)

        self.is_status = False
        self.is_obj = False
        self.is_pose_received = False
        self.object_pose = None
        self.is_path_received = False
        self.object_path = None
        self.deleted_ids = set()
        self.is_lattice_path = False

        self.time_gap = 1.5

    def control_velocity(self, target_velocity):

        desired_velocity = target_velocity

        is_forward, forward_dist, forward_speed = self.forward_vehicle(self.lattice_path, self.object_data)
        if is_forward:
            if forward_dist < self.time_gap * self.status_msg.velocity.x:
                desired_velocity = min(desired_velocity, forward_speed - 2.0)
        elif self.checkObject_npc_path(self.lattice_path, self.object_path):
            desired_velocity = target_velocity - 5.0

        return desired_velocity


    def transform_to_local(self, global_position, reference_position, reference_theta):
        translation = np.array([global_position.x - reference_position.x,
                                global_position.y - reference_position.y])
        rotation_matrix = np.array([[cos(-reference_theta), -sin(-reference_theta)],
                                    [sin(-reference_theta), cos(-reference_theta)]])
        local_position = rotation_matrix.dot(translation)
        return Point(x=local_position[0], y=local_position[1], z=0)

    def forward_vehicle(self, ref_path, object_data):

        forward_vehicle = ref_path.poses[0].pose.position
        forward_theta = atan2(ref_path.poses[1].pose.position.y - forward_vehicle.y,
                      ref_path.poses[1].pose.position.x - forward_vehicle.x)

        local_path = [self.transform_to_local(pose.pose.position, forward_vehicle, forward_theta) for pose in ref_path.poses]

        for local_pose in local_path:
            for npc in object_data.npc_list:
                local_npc_position = self.transform_to_local(npc.position, forward_vehicle, forward_theta)
                if 0 < (local_npc_position.x - local_pose.x) < 50 and abs(local_npc_position.y - local_pose.y) < 1.75:
                    return True, local_npc_position.x - local_pose.x, npc.velocity.x

        return False, 0
    
    def checkObject_npc_path(self, ref_path, object_path):
        
        def is_path_overlap(path_pose, predicted_pose, threshold):
            dis = sqrt(pow(path_pose.x - predicted_pose.x, 2) + pow(path_pose.y - predicted_pose.y, 2))
            return dis < threshold

        vehicle_position = ref_path.poses[0].pose.position
        theta = atan2(ref_path.poses[1].pose.position.y - vehicle_position.y,
                      ref_path.poses[1].pose.position.x - vehicle_position.x)

        local_path = [self.transform_to_local(pose.pose.position, vehicle_position, theta) for pose in ref_path.poses]

        # npc's predicted path
        if object_path is not None:
            for local_pose in local_path:
                for predicted_path in object_path.path_list:
                    for predicted_pose in predicted_path.path:
                        local_predicted_position = self.transform_to_local(Point(x=predicted_pose.x, y=predicted_pose.y, z=0), vehicle_position, theta)
                        if is_path_overlap(local_pose, local_predicted_position, 2.35):
                            return True
        return False    
    
    
    
    def status_callback(self, msg):  ## Vehicle Status Subscriber
        self.is_status = True
        self.status_msg = msg

    def object_callback(self, msg):
        self.is_obj = True
        self.object_data = msg

    def object_info_callback(self, msg):
        self.is_pose_received = True
        self.object_pose = msg

    def object_path_callback(self, msg):
        self.is_path_received = True
        self.object_path = msg

    def deleted_object_callback(self, msg):
        self.deleted_ids.add(msg.data)
    
    def lattice_path_callback(self, msg):
        self.is_lattice_path = True
        self.lattice_path = msg

