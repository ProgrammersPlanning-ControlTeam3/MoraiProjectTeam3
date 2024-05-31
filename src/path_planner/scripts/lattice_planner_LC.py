#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import rospy
from math import cos, sin, pi, sqrt, pow, atan2
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList
from prediction.msg import TrackedPoint, PredictedObjectPath, PredictedObjectPathList, TrackedObjectPose, TrackedObjectPoseList
from geometry_msgs.msg import Point, PoseStamped, Point32
from nav_msgs.msg import Path
from std_msgs.msg import Int32

import numpy as np
from frame_transform import *

class latticePlanner:
    def __init__(self):
        rospy.init_node(name='lattice_planner', anonymous=True)

        # (1) subscriber, publisher 선언
        rospy.Subscriber("/local_path", Path, self.path_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_callback)
        rospy.Subscriber('/Object_topic/tracked_object_pose_topic', TrackedObjectPoseList, self.object_info_callback)
        rospy.Subscriber('/Object_topic/tracked_object_path_topic', PredictedObjectPathList, self.object_path_callback)
        rospy.Subscriber('/Object_topic/deleted_object_id', Int32, self.deleted_object_callback)

        self.lattice_path_pub = rospy.Publisher('/lattice_path', Path, queue_size=1)

        self.is_path = False
        self.is_status = False
        self.is_obj = False
        self.is_pose_received = False
        self.object_pose = None
        self.is_path_received = False
        self.object_path = None
        self.deleted_ids = set()

        self.foward_vehicle_speed = 0
        self.target_velocity = 40  # Target Velocity in m/s
        self.local_path_size = 30

        rate = rospy.Rate(30)  # 30hz
        while not rospy.is_shutdown():
            if self.is_path and self.is_status and self.is_obj:
                if self.checkObject(self.local_path, self.object_data, self.object_path):
                    self.foward_vehicle_speed = self.get_forward_vehicle(self.local_path, self.object_data)
                    lattice_path = self.latticePlanner(self.local_path, self.status_msg)
                    lattice_path_index = self.collision_check(self.object_data, lattice_path)
                    # (7)  lattice 경로 메세지 Publish
                    self.lattice_path_pub.publish(lattice_path[lattice_path_index])
                else:
                    self.lattice_path_pub.publish(self.local_path)
            rate.sleep()


    def transform_to_local(self, global_position, reference_position, reference_theta):
        translation = np.array([global_position.x - reference_position.x,
                                global_position.y - reference_position.y])
        rotation_matrix = np.array([[cos(-reference_theta), -sin(-reference_theta)],
                                    [sin(-reference_theta), cos(-reference_theta)]])
        local_position = rotation_matrix.dot(translation)
        return Point(x=local_position[0], y=local_position[1], z=0)


    def get_forward_vehicle(self, ref_path, object_data):

        forward_vehicle = ref_path.poses[0].pose.position
        forward_theta = atan2(ref_path.poses[1].pose.position.y - forward_vehicle.y,
                      ref_path.poses[1].pose.position.x - forward_vehicle.x)

        local_path = [self.transform_to_local(pose.pose.position, forward_vehicle, forward_theta) for pose in ref_path.poses]

        for local_pose in local_path:
            for npc in object_data.npc_list:
                local_npc_position = self.transform_to_local(npc.position, forward_vehicle, forward_theta)
                if 0 < (local_npc_position.x - local_pose.x) < 30 and abs(local_npc_position.y - local_pose.y) < 1:
                    # print("Vehicle ahead : ", local_npc_position.x - local_pose.x)
                    # print(npc.velocity.x)
                    return npc.velocity.x

        return None


    def checkObject(self, ref_path, object_data, object_path):
        def is_collision_distance(path_pose, obj_position, threshold):
            dis = sqrt(pow(path_pose.x - obj_position.x, 2) + pow(path_pose.y - obj_position.y, 2))
            return dis < threshold

        def is_path_overlap(path_pose, predicted_pose, threshold):
            dis = sqrt(pow(path_pose.x - predicted_pose.x, 2) + pow(path_pose.y - predicted_pose.y, 2))
            return dis < threshold

        vehicle_position = ref_path.poses[0].pose.position
        theta = atan2(ref_path.poses[1].pose.position.y - vehicle_position.y,
                      ref_path.poses[1].pose.position.x - vehicle_position.x)

        local_path = [self.transform_to_local(pose.pose.position, vehicle_position, theta) for pose in ref_path.poses]

        # npc's position
        for local_pose in local_path:
            for npc in object_data.npc_list:
                local_npc_position = self.transform_to_local(npc.position, vehicle_position, theta)
                if is_collision_distance(local_pose, local_npc_position, 2.35):
                    print("NPC")
                    return True

        # npc's predicted path
        if object_path is not None:
            for local_pose in local_path:
                for predicted_path in object_path.path_list:
                    for predicted_pose in predicted_path.path:
                        local_predicted_position = self.transform_to_local(Point(x=predicted_pose.x, y=predicted_pose.y, z=0), vehicle_position, theta)
                        if is_path_overlap(local_pose, local_predicted_position, 2.35):
                            print("Path")
                            return True
        return False


    def collision_check(self, object_data, out_path):

        def cost_function(dist, i):
            cost = -(dist - i)**2 + 20
            return cost

        # TODO: (6) 생성된 충돌회피 경로 중 낮은 비용의 경로 선택
        selected_lane = -1
        lane_weight = [3,2,1,3,2,1]

        for obstacle in object_data.npc_list:
            for path_num in range(len(out_path)):
                for path_pos in out_path[path_num].poses:

                    dis = sqrt(
                        pow(obstacle.position.x - path_pos.pose.position.x, 2) +
                        pow(obstacle.position.y - path_pos.pose.position.y, 2)
                    )

                    if dis < 4:
                        centre = 0
                    elif dis < 8:
                        centre = 1
                    elif dis < 12:
                        centre = 2
                    elif dis < 16:
                        centre = 3
                    elif dis < 20:
                        centre = 4
                    else:
                        centre = 5

                    lane_weight[path_num] += cost_function(centre, path_num)


        selected_lane = lane_weight.index(max(lane_weight))
        print("Lane change : ", selected_lane)

        return selected_lane



    # def collision_check(self, object_data, out_path):
    #     # TODO: (6) 생성된 충돌회피 경로 중 낮은 비용의 경로 선택
    #     selected_lane = -1
    #     lane_weight = [11, 12, 13, 1, 2, 3]


    #     # TODO remove comment
    #     # path_size = self.foward_vehicle_speed
    #     path_size = 50

    #     # if self.foward_vehicle_speed is None:
    #     #     path_size = self.local_path_size

    #     short_path_size = path_size * 0.6

    #     for obstacle in object_data.npc_list:
    #         for path_num in range(len(out_path)):
    #             for path_pos in out_path[path_num].poses:
    #                 dis = sqrt(
    #                     pow(obstacle.position.x - path_pos.pose.position.x, 2) + pow(obstacle.position.y - path_pos.pose.position.y, 2))

    #                 # apply weight to all paths
    #                 if dis < short_path_size:
    #                     if path_num < 3:
    #                         weight_increase = 10
    #                     else:
    #                         weight_increase = 30
    #                 # weight to 3, 4, 5 paths only
    #                 elif short_path_size <= dis < path_size:
    #                     if path_num < 3:
    #                         weight_increase = 30
    #                     else:
    #                         weight_increase = 0
    #                 else:
    #                     weight_increase = 0

    #                 lane_weight[path_num] += weight_increase


    #     selected_lane = lane_weight.index(min(lane_weight))
    #     print("Lane change : ", selected_lane)

    #     return selected_lane

    def path_callback(self, msg):
        self.is_path = True
        self.local_path = msg

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

    def latticePlanner(self, ref_path, vehicle_status):
        out_path = []
        vehicle_pose_x = vehicle_status.position.x
        vehicle_pose_y = vehicle_status.position.y
        vehicle_velocity = vehicle_status.velocity.x * 3.6

        look_distance = int(vehicle_velocity * 0.2 * 2)

        if look_distance < 20:
            look_distance = 20

        # ref_path.poses의 길이를 고려하여 look_distance 조정
        max_look_distance = len(ref_path.poses) // 2 - 1
        look_distance = min(look_distance, max_look_distance)

        if len(ref_path.poses) > look_distance:
            # 지도 데이터
            mapx = [pose.pose.position.x for pose in ref_path.poses]
            mapy = [pose.pose.position.y for pose in ref_path.poses]
            maps = [0]
            for i in range(1, len(mapx)):
                maps.append(maps[-1] + get_dist(mapx[i-1], mapy[i-1], mapx[i], mapy[i]))

            global_ref_end_point = (ref_path.poses[look_distance * 2].pose.position.x,
                                    ref_path.poses[look_distance * 2].pose.position.y)

            vehicle_s, vehicle_d = get_frenet(vehicle_pose_x, vehicle_pose_y, mapx, mapy)

            goal_s, goal_d = get_frenet(global_ref_end_point[0], global_ref_end_point[1], mapx, mapy)
            lane_offsets = [5, 4.25, 3.5]
            time_offsets = [0.6, 1.0]

            for time_offset in time_offsets:
                for lane_offset in lane_offsets:
                    lattice_path = Path()
                    lattice_path.header.frame_id = 'map'
                    goal_d_with_offset = vehicle_d + lane_offset 

                    # TODO remove comment
                    # forward vehicle's speed based target point
                    # if self.foward_vehicle_speed is not None:
                    #     goal_s_with_offset = vehicle_s + min(self.target_velocity, self.foward_vehicle_speed) * time_offset
                    # else :
                    #     goal_s_with_offset = vehicle_s + self.target_velocity * time_offset
                    goal_s_with_offset = vehicle_s + 50 * time_offset


                    # 5차 곡선
                    xs = 0
                    xf = goal_s_with_offset - vehicle_s
                    ys = vehicle_d
                    yf = goal_d_with_offset

                    a = self.generate_5th_order_polynomial(ys, yf, xs, xf)
                    x_vals = np.linspace(xs, xf, num=100)
                    y_vals = self.calculate_polynomial(a, x_vals)

                    for i in range(len(x_vals)):
                        x = x_vals[i] + vehicle_s
                        y = y_vals[i]

                        global_x, global_y, _ = get_cartesian(x, y, mapx, mapy, maps)
                        
                        read_pose = PoseStamped()
                        read_pose.pose.position.x = global_x
                        read_pose.pose.position.y = global_y
                        read_pose.pose.position.z = 0
                        read_pose.pose.orientation.x = 0
                        read_pose.pose.orientation.y = 0
                        read_pose.pose.orientation.z = 0
                        read_pose.pose.orientation.w = 1
                        lattice_path.poses.append(read_pose)

                    out_path.append(lattice_path)

            # Add_point
            add_point_size = min(int(vehicle_velocity * 2), len(ref_path.poses))

            for i in range(look_distance * 2, add_point_size):
                if i + 1 < len(ref_path.poses):
                    tmp_theta = atan2(
                        ref_path.poses[i + 1].pose.position.y - ref_path.poses[i].pose.position.y,
                        ref_path.poses[i + 1].pose.position.x - ref_path.poses[i].pose.position.x)
                    tmp_translation = [ref_path.poses[i].pose.position.x, ref_path.poses[i].pose.position.y]
                    tmp_t = np.array(
                        [[cos(tmp_theta), -sin(tmp_theta), tmp_translation[0]], [sin(tmp_theta), cos(tmp_theta), tmp_translation[1]], [0, 0, 1]])

                    for lane_num in range(len(lane_offsets)):
                        local_result = np.array([[0], [lane_offsets[lane_num]], [1]])
                        global_result = tmp_t.dot(local_result)

                        read_pose = PoseStamped()
                        read_pose.pose.position.x = global_result[0][0]
                        read_pose.pose.position.y = global_result[1][0]
                        read_pose.pose.position.z = 0
                        read_pose.pose.orientation.x = 0
                        read_pose.pose.orientation.y = 0
                        read_pose.pose.orientation.z = 0
                        read_pose.pose.orientation.w = 1
                        out_path[lane_num].poses.append(read_pose)

            # 생성된 모든 Lattice 충돌 회피 경로 메시지 Publish
            for i in range(len(out_path)):
                globals()['lattice_pub_{}'.format(i + 1)] = rospy.Publisher('/lattice_path_{}'.format(i + 1), Path, queue_size=1)
                globals()['lattice_pub_{}'.format(i + 1)].publish(out_path[i])

        return out_path


if __name__ == '__main__':
    try:
        latticePlanner()
    except rospy.ROSInterruptException:
        pass
