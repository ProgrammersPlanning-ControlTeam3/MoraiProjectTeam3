#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import rospy
from math import cos, sin, pi, sqrt, pow, atan2
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList
from prediction.msg import TrackedPoint, PredictedObjectPath, PredictedObjectPathList, TrackedObjectPose, TrackedObjectPoseList, PredictedHMM, PredictedHMMPath
from geometry_msgs.msg import Point, PoseStamped, Point32
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Int32

import numpy as np
from frame_transform import *   

isInParkingLot= False

class latticePlanner:
    def __init__(self):
        rospy.init_node(name='lattice_planner', anonymous=True)

        # (1) subscriber, publisher 선언
        rospy.Subscriber("/local_path", Path, self.path_callback)
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_callback)
        rospy.Subscriber('/Object_topic/deleted_object_id', Int32, self.deleted_object_callback)
        rospy.Subscriber('/Object_topic/hmm_prediction', PredictedHMM, self.prediction_info_callback)
        rospy.Subscriber('/Object_topic/hmm_predicted_path_topic', PredictedHMMPath, self.prediction_path_callback)

        self.lattice_path_pub = rospy.Publisher('/lattice_path', Path, queue_size=1)
        self.selected_lane_pub = rospy.Publisher('/selected_lane', Int32, queue_size=1)

        self.is_path = False
        self.is_status = False
        self.is_obj = False
        self.is_path_received = False
        self.is_prediction_received = False
        self.is_prediction_path_received = False
        self.deleted_ids = set()
        self.prediction_path = None
        self.vehicle_paths = {}

        self.vehicle_probability_LK = 0.0
        self.vehicle_probability_LR = 0.0
        self.vehicle_probability_LL = 0.0

        self.foward_vehicle_speed = 0
        self.target_velocity = 40  # Target Velocity in m/s
        self.local_path_size = 30

        rate = rospy.Rate(30)  # 30hz
        while not rospy.is_shutdown():
            if self.is_path and self.is_status and self.is_obj:
                forward_vehicle = self.get_forward_vehicle(self.local_path, self.object_data)
                if forward_vehicle is not None:
                    self.foward_vehicle_speed = forward_vehicle.velocity.x

                    # if self.is_prediction_received and self.is_prediction_path_received:
                lattice_path = self.latticePlanner(self.local_path, self.x, self.y)
                lattice_path_index = self.collision_check(self.object_data, lattice_path)

                # (7)  lattice 경로 메세지 Publish
                self.lattice_path_pub.publish(lattice_path[lattice_path_index])
                self.selected_lane_pub.publish(Int32(data=lattice_path_index))
            rate.sleep()


    def predicted(self):
        if self.prediction_path is None:
            rospy.logwarn("Prediction path is not received yet.")
            return

        vehicle_manuever = self.prediction_data.maneuver

        ## probability ##
        vehicle_maneuvers = self.prediction_data.probability
        vehicle_maneuvers_first = vehicle_maneuvers[0]
        self.vehicle_probability_LK = vehicle_maneuvers_first.lane_keeping
        self.vehicle_probability_LR = vehicle_maneuvers_first.right_change
        self.vehicle_probability_LL = vehicle_maneuvers_first.left_change

        ## path ##
        vehicle_path_LK = self.prediction_path.lane_keeping_path
        vehicle_path_LR = self.prediction_path.right_change_path
        vehicle_path_LL = self.prediction_path.left_change_path

        self.vehicle_paths = {
            "lane_keeping": (vehicle_path_LK, self.vehicle_probability_LK),
            "right_change": (vehicle_path_LR, self.vehicle_probability_LR),
            "left_change": (vehicle_path_LL, self.vehicle_probability_LL)
        }


    def transform_to_local(self, global_position, reference_position, reference_theta):
        translation = np.array([global_position.x - reference_position.x,
                                global_position.y - reference_position.y])
        rotation_matrix = np.array([[cos(-reference_theta), -sin(-reference_theta)],
                                    [sin(-reference_theta), cos(-reference_theta)]])
        local_position = rotation_matrix.dot(translation)
        return Point(x=local_position[0], y=local_position[1], z=0)


    def get_forward_vehicle(self, ref_path, object_data):
        if len(ref_path.poses) < 2:
            return None

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
                    return npc
        return None



    def get_npc_ids_in_range(self, ref_path, object_data, x_min, x_max, y_min, y_max):
        forward_vehicle = ref_path.poses[0].pose.position
        forward_theta = atan2(ref_path.poses[1].pose.position.y - forward_vehicle.y,
                            ref_path.poses[1].pose.position.x - forward_vehicle.x)

        npc_ids_in_range = []
        for npc in object_data.npc_list:
            local_npc_position = self.transform_to_local(npc.position, forward_vehicle, forward_theta)
            if x_min <= local_npc_position.x <= x_max and y_min <= local_npc_position.y <= y_max:
                npc_ids_in_range.append(npc.unique_id)
        return npc_ids_in_range


    def collision_check(self, object_data, out_path):
        selected_lane = -1
        lane_weight = [5, 2, 1000, 5, 0, 10000]
        maneuver_weights = [{"lane_keeping": 0, "right_change": 0, "left_change": 0} for _ in range(len(out_path))]

        def calculate_risk(center1, center2):
            distance_squared = pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2)
            distance = pow(distance_squared, 0.5)
            risk = 1 / distance_squared if distance_squared != 0 else float('inf')
            if distance <= 0.5:
                risk += 1
            return risk

        # Forward vehicle check
        forward_vehicle_check = self.get_forward_vehicle(self.local_path, object_data)
        forward_vehicle_id = None
        if forward_vehicle_check is not None:
            if self.foward_vehicle_speed > 15:
                path_length = min(self.target_velocity, self.status_msg.velocity.x * 3.0)
            else:
                path_length = self.target_velocity
            short_path_length = path_length * 0.6
            print(path_length)

        npc_ids_in_range = self.get_npc_ids_in_range(self.local_path, object_data, -40, 50, -5, 5)

        if npc_ids_in_range:
            self.predicted()

            for path_num in range(len(out_path)):
                path_len = len(out_path[path_num].poses)
                for npc_id in npc_ids_in_range:
                    for predicted_path_key, (global_predicted_path_list, prob) in self.vehicle_paths.items():
                        total_risk = 0
                        npc_len = len(global_predicted_path_list)
                        for n in range(min(path_len, npc_len)):  # 길이 비교
                            vehicle_circle_center = out_path[path_num].poses[n].pose.position
                            npc_circle_center = global_predicted_path_list[n]
                            npc_point = Point(x=npc_circle_center.x, y=npc_circle_center.y, z=0)

                            risk = calculate_risk(vehicle_circle_center, npc_point)
                            total_risk += risk

                        maneuver_weights[path_num][predicted_path_key] += total_risk

            # 각 경로에 대한 총 weight를 확률을 곱하여 lane_weight에 반영
            for path_num in range(len(out_path)):
                lane_weight[path_num] += maneuver_weights[path_num]["lane_keeping"] * self.vehicle_probability_LK
                lane_weight[path_num] += maneuver_weights[path_num]["right_change"] * self.vehicle_probability_LR
                lane_weight[path_num] += maneuver_weights[path_num]["left_change"] * self.vehicle_probability_LL

                print("[", path_num, "LK ] : ", maneuver_weights[path_num]["lane_keeping"])
                print("[", path_num, "LR ] : ", maneuver_weights[path_num]["right_change"])
                print("[", path_num, "LL ] : ", maneuver_weights[path_num]["left_change"])
                print("\n")

        if forward_vehicle_check is not None and npc_ids_in_range:
            # Calculate distance between the first points
            if "lane_keeping" in self.vehicle_paths and len(self.vehicle_paths["lane_keeping"][0]) > 0:
                vehicle_circle_center = out_path[0].poses[0].pose.position
                npc_circle_center = self.vehicle_paths["lane_keeping"][0][0]
                npc_point = Point(x=npc_circle_center.x, y=npc_circle_center.y, z=0)

                distance = pow(pow(vehicle_circle_center.x - npc_point.x, 2) + pow(vehicle_circle_center.y - npc_point.y, 2), 0.5)

                if distance < short_path_length:
                    selected_lanes = [0, 1, 2]
                else:
                    selected_lanes = [3, 4, 5]

                lane_weight_selected = {lane: lane_weight[lane] for lane in selected_lanes}
                selected_lane = min(lane_weight_selected, key=lane_weight_selected.get)
        else:
            selected_lane = lane_weight.index(min(lane_weight))

        print("Lane change : ", selected_lane)
        print("min weight : ", lane_weight[selected_lane])
        print("0 : ", lane_weight[0])
        print("1 : ", lane_weight[1])
        print("2 : ", lane_weight[2])
        print("3 : ", lane_weight[3])
        print("4 : ", lane_weight[4])
        print("5 : ", lane_weight[5])
        print("\n")

        if lane_weight[selected_lane] >= 1000: # 모든 경로의 웨이트가 높은 경우 직진 -> 감속하기 위해서
            selected_lane = 1
            # print("Lane change : ", selected_lane)

        return selected_lane





    def path_callback(self, msg):
        self.is_path = True
        self.local_path = msg

    def status_callback(self, msg):  ## Vehicle Status Subscriber
        self.is_status = True
        self.status_msg = msg

    def odom_callback(self, msg):
        self.is_status=True
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y


    def object_callback(self, msg):
        self.is_obj = True
        self.object_data = msg

    def deleted_object_callback(self, msg):
        self.deleted_ids.add(msg.data)

    def prediction_info_callback(self, msg):
        self.is_prediction_received = True
        self.prediction_data = msg
        # unique_id, maneuver: "Lane Keeping", probability[lane_keeping, right_change, left_change]

    def prediction_path_callback(self, msg):
        self.is_prediction_path_received = True
        self.prediction_path = msg
        # unique_id, lane_keeping_path, right_change_path, left_change_path

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

    def latticePlanner(self, ref_path, x, y):
        out_path = []
        vehicle_pose_x = x
        vehicle_pose_y = y
        vehicle_velocity = self.status_msg.velocity.x * 3.6

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

            lane_offsets = [3.5, 0, -3.5]
            time_offsets = [0.6, 1.0]

            for time_offset in time_offsets:
                for lane_offset in lane_offsets:
                    lattice_path = Path()
                    lattice_path.header.frame_id = 'map'
                    goal_d_with_offset = vehicle_d + lane_offset

                    # forward vehicle's speed based target point -> changed to controlled velocity (전방향 차량 속도에 따라 제어된 속도값 사용 : 현재 차량의 속도값 사용하게 됨)
                    if self.foward_vehicle_speed > 10:
                        goal_s_with_offset = vehicle_s + min(self.target_velocity, max(self.status_msg.velocity.x * 3.0, 30)) * time_offset
                    else :
                        goal_s_with_offset = vehicle_s + self.target_velocity * time_offset

                    # Test용 코드
                    # goal_s_with_offset = vehicle_s + 50 * time_offset


                    # 5차 곡선
                    xs = vehicle_s
                    xf = goal_s_with_offset
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
