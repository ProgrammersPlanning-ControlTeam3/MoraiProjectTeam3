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

        rate = rospy.Rate(30)  # 30hz
        while not rospy.is_shutdown():
            if self.is_path and self.is_status and self.is_obj:
                if self.checkObject(self.local_path, self.object_data, self.object_path):
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
        # TODO: (6) 생성된 충돌회피 경로 중 낮은 비용의 경로 선택

        selected_lane = -1
        lane_weight = [1, 10, 100]  # reference path

        for obstacle in object_data.npc_list:
            for path_num in range(len(out_path)):
                for path_pos in out_path[path_num].poses:
                    dis = sqrt(
                        pow(obstacle.position.x - path_pos.pose.position.x, 2) + pow(obstacle.position.y - path_pos.pose.position.y, 2))

                    # weight based on distance
                    if dis < 10:
                        weight_increase = 20
                    elif dis < 25:
                        weight_increase = 10
                    elif dis < 40:
                        weight_increase = 5
                    else:
                        weight_increase = 0
                    lane_weight[path_num] += weight_increase

        selected_lane = lane_weight.index(min(lane_weight))
        print("Lane change : ", selected_lane)

        return selected_lane

    def path_callback(self, msg):
        self.is_path = True
        self.local_path = msg

    def status_callback(self, msg):  ## Vehicl Status Subscriber
        self.is_status = True
        self.status_msg = msg

    def object_callback(self, msg):
        self.is_obj = True
        self.object_data = msg

    def object_info_callback(self, msg):
        self.is_pose_received=True
        self.object_pose = msg

    def object_path_callback(self, msg):
        self.is_path_received=True
        self.object_path = msg

    def deleted_object_callback(self, msg):
        self.deleted_ids.add(msg.data)

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
            # TODO: (3) 좌표 변환 행렬 생성
            """
            # 좌표 변환 행렬을 만듭니다.
            # Lattice 경로를 만들기 위해서 경로 생성을 시작하는 Point 좌표에서 
            # 경로 생성이 끝나는 Point 좌표의 상대 위치를 계산해야 합니다.
            """

            global_ref_start_point = (ref_path.poses[0].pose.position.x, ref_path.poses[0].pose.position.y)
            global_ref_start_next_point = (ref_path.poses[1].pose.position.x, ref_path.poses[1].pose.position.y)

            global_ref_end_point = (ref_path.poses[look_distance * 2].pose.position.x,
                                    ref_path.poses[look_distance * 2].pose.position.y)

            theta = atan2(global_ref_start_next_point[1] - global_ref_start_point[1],
                          global_ref_start_next_point[0] - global_ref_start_point[0])
            translation = [global_ref_start_point[0], global_ref_start_point[1]]

            trans_matrix = np.array([[cos(theta), -sin(theta), translation[0]],
                                     [sin(theta), cos(theta), translation[1]],
                                     [0, 0, 1]])

            det_trans_matrix = np.array([[trans_matrix[0][0], trans_matrix[1][0],
                                          -(trans_matrix[0][0] * translation[0] + trans_matrix[1][0] * translation[1])],
                                         [trans_matrix[0][1], trans_matrix[1][1],
                                          -(trans_matrix[0][1] * translation[0] + trans_matrix[1][1] * translation[1])],
                                         [0, 0, 1]])

            world_end_point = np.array([[global_ref_end_point[0]], [global_ref_end_point[1]], [1]])
            local_end_point = det_trans_matrix.dot(world_end_point)
            world_ego_vehicle_position = np.array([[vehicle_pose_x], [vehicle_pose_y], [1]])
            local_ego_vehicle_position = det_trans_matrix.dot(world_ego_vehicle_position)
            lane_off_set = [0, 2, 4]  # Only 3 offsets now
            local_lattice_points = []

            for i in range(len(lane_off_set)):
                local_lattice_points.append([local_end_point[0][0], local_end_point[1][0] + lane_off_set[i], 1])

            # TODO: (4) Lattice 충돌 회피 경로 생성
            '''
            # Local 좌표계로 변경 후 3차곡선계획법에 의해 경로를 생성한 후 다시 Map 좌표계로 가져옵니다.
            # Path 생성 방식은 3차 방정식을 이용하며 lane_change_ 예제와 동일한 방식의 경로 생성을 하면 됩니다.
            # 생성된 Lattice 경로는 out_path 변수에 List 형식으로 넣습니다.
            # 충돌 회피 경로는 기존 경로를 제외하고 좌 우로 1개씩 총 2개의 경로를 가지도록 합니다.
            '''

            for end_point in local_lattice_points:
                lattice_path = Path()
                lattice_path.header.frame_id = 'map'
                x = []
                y = []
                x_interval = 0.5
                xs = 0
                xf = end_point[0]
                ps = local_ego_vehicle_position[1][0]

                pf = end_point[1]
                x_num = xf / x_interval

                for i in range(xs, int(x_num)):
                    x.append(i * x_interval)

                a = [0.0, 0.0, 0.0, 0.0]
                a[0] = ps
                a[1] = 0
                a[2] = 3.0 * (pf - ps) / (xf * xf)
                a[3] = -2.0 * (pf - ps) / (xf * xf * xf)

                # 3차 곡선 계획
                for i in x:
                    result = a[3] * i * i * i + a[2] * i * i + a[1] * i + a[0]
                    y.append(result)

                for i in range(0, len(y)):
                    local_result = np.array([[x[i]], [y[i]], [1]])
                    global_result = trans_matrix.dot(local_result)

                    read_pose = PoseStamped()
                    read_pose.pose.position.x = global_result[0][0]
                    read_pose.pose.position.y = global_result[1][0]
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

                    for lane_num in range(len(lane_off_set)):
                        local_result = np.array([[0], [lane_off_set[lane_num]], [1]])
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

            # TODO: (5) 생성된 모든 Lattice 충돌 회피 경로 메시지 Publish
            '''
            # 생성된 모든 Lattice 충돌회피 경로는 ros 메세지로 송신하여
            # Rviz 창에서 시각화 하도록 합니다.
            '''
            for i in range(len(out_path)):
                globals()['lattice_pub_{}'.format(i + 1)] = rospy.Publisher('/lattice_path_{}'.format(i + 1), Path, queue_size=1)
                globals()['lattice_pub_{}'.format(i + 1)].publish(out_path[i])

        return out_path


if __name__ == '__main__':
    try:
        latticePlanner()
    except rospy.ROSInterruptException:
        pass
