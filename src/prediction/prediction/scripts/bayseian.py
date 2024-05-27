#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import copy
import bisect
import matplotlib.cm as cm
import matplotlib.animation as animation
import bisect

from IPython.display import HTML
from utils import *
from agent import agent
from filter import *



import tf
import rospkg
import rospy

from geometry_msgs.msg import Twist, Point32, PolygonStamped, Polygon, Vector3, Pose, Quaternion, Point
from visualization_msgs.msg import MarkerArray, Marker

from std_msgs.msg import Float32, Float64, Header, ColorRGBA, UInt8, String, Float32MultiArray, Int32MultiArray

# np.random.seed(700 )
class DynamicObstacleTracker:
    def __init__(self, dt=0.1, T=1):
        self.vehicles={}
        self.int_pt_list={}
        self.dt=dt
        self.initialize()

    def initialize(self):
        self.TrackingList={} # Tracking Vehicle Lists
        self.PossiblePath={} # Tracking Vehicle Path Info Lists
        
        self.OccupiedPt={}
        self.FilteredHistory = {}
        self.Filter={}
    
    def Tracking(self, data):
        """

        1. Get Global Coordinate Info form MORAI.
        2. Added Extended Kalman Filter to "Filter" for filtering
        3. Conduct filtering using current measured information
        4. If not detected, count number, and if reached, eliminate.

        Returns:
            _type_: _description_
        """
        #TODO(1). Get Global Information from MORAI
        #rospy.subscriber("/Object_topic", ObjectStatusList, self.object_info_callback)
        global_info = data # data is morai info. You have to subscribe the messages
        '''
            `ObjectStatus` `{`

            `int32 unique_id`

            `int32 type`

            `string name`

            `float64 heading`

            `geometry_msgs/Vector3 velcotiy`

            `Acceleartion`

            `Size`

            `position` 

            `}`
        '''
        if len(global_info) == 0:
            pass
        else:
            for measured_vehicle  in global_info:
                id=measured_vehicle[0]
                x= measured_vehicle[8].x
                y= measured_vehicle[8].y
                h= measured_vehicle[3]
                vx= measured_vehicle[4].vx
                vy= measured_vehicle[4].vy
                #TODO(2): Convert Coordinate from Cartesian to Frenet
                # Define Lane Width
    def Maneuver_pred(self):
        self.OccupiedPt={}
        for agent_id in self.FitleredHistory.keys()
class DynamicObstacleTrackerNode:
    def __init__(self):
        rospy.init_node("dynamic_obstacle_tracker_node", anonymous=True)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_info_callback)

    def object_info_callback(self, msg):
        rospy.loginfo("Received Object Message")
        self.is_object=True
        self.object_data = msg

class Environments(object):
    def __init__(self, course_idx, dt=0.1, min_num_agent=8):
                
        self.spawn_id = 0
        self.vehicles = {}
        self.int_pt_list = {}
        self.min_num_agent = min_num_agent
        self.dt = dt
        self.course_idx = course_idx

        self.initialize()

    def initialize(self, init_num=6):
        
        self.pause = False
        filepath = rospy.get_param("file_path")
        Filelist = glob.glob(filepath+"/*info.pickle")
        
        file = Filelist[0]

        with open(file, "rb") as f:
            Data = pickle.load(f)
            
        self.map_pt = Data["Map"]
        self.connectivity = Data["AS"]
        
        for i in range(init_num):
            if i==0:
                CourseList = [[4,1,18], [4,2,25], [4,0,11]]
                self.spawn_agent(target_path = CourseList[self.course_idx], init_v = 0)
            else:
                self.spawn_agent()

        ### For visualization ###
        self.TrackingList = {}
        self.PossiblePath = {}
        
        ### For longitudinal control ###
        self.map_pt_list = np.concatenate([self.map_pt[key][:-1,:3] for key in self.map_pt.keys()], axis=0)
        self.map_pt_idx = np.cumsum([len(self.map_pt[key][:-1,:]) for key in self.map_pt.keys()], axis=0)
        self.OccupiedPt = {}
        
        ### For object history tracking ###
        self.FilteredHistory = {}
        self.Filter = {}
        self.DelCnt = np.zeros([100])

    def delete_agent(self):

        delete_agent_list = []

        for id_ in self.vehicles.keys():
            if (self.vehicles[id_].target_s[-1]-10) < self.vehicles[id_].s:
                delete_agent_list.append(id_)

        return delete_agent_list
    
    def Tracking(self, sensor_info):

        ##############################
        ### agent history tracking ###
        ##############################
        
        """
        step1. sensor에서 받은 Local 좌표를 자 차량 정보를 활용하여 Global 좌표로 수정
        step2. Detect된 object 중 처음 detect된 경우, filtering을 위한 EKF를 "Filter"에 추가
        step3. 이미 detect된 적이 있는 경우에는 현재 정보를 measure로 filtering 수행 
        step4. 관리하고 있는 object 중 현재 시점에 detect가 안된 경우 count를 올리고, 일정 count가 되면 관리 목록에서 제거
        """
        
        global_info = self.vehicles[0].ToGlobal(sensor_info)                               ### step 1 ###
                
        if len(global_info) == 0:
            pass
        
        else:
            for measured_agent in global_info:
                id = measured_agent[0]
                x = measured_agent[1]
                y = measured_agent[2]
                h = measured_agent[3]
                vx = measured_agent[4]
                vy = measured_agent[5]
                
                if id in self.FilteredHistory:                                            ### step 2 & 3 ###
                    if len(self.FilteredHistory[id])>10:
                        self.FilteredHistory[id].pop(0)    
                        self.TrackingList[id].pop(0)
                    
                    self.Filter[id].predict(Q=np.diag([0.001, 0.001, 0.1, 0.1, 0.01, 0.01]))    
                    z = [x, y, (vx**2+vy**2)**0.5, h]
                    self.Filter[id].correction(z = z, R=np.diag([100, 100, 1,  0.1]))
                    
                else:
                    self.FilteredHistory[id] = []
                                   
                    state = [x, y, (vx**2+vy**2)**0.5, 0, h, 0]
                    self.Filter[id] = init_kf(state)

                    self.TrackingList[id] = []     
                    
                self.FilteredHistory[id].append(self.Filter[id].x)
                self.TrackingList[id].append([x,y,h,vx,vy])  
                
            
            del_list = []                                                                  ### step 4 ###
            for key in self.FilteredHistory.keys():
                measured_id_list = global_info[:,0]
                
                if key in measured_id_list:
                    pass
                else:
                    self.DelCnt[int(key)]+=1
                
                if self.DelCnt[int(key)]>1:
                    del_list.append(key)
                    
            [self.FilteredHistory.pop(key, None) for key in del_list]
            [self.Filter.pop(key, None) for key in del_list]
            [self.TrackingList.pop(key, None) for key in del_list]
            
            
    
    def Maneuver_pred(self):
        
        ##############################
        ##### Maenuver Prediction #### 
        ##############################
        
        """
        step1. 맵 정보와 filtering된 object 정보를 통해 현재 차선 및 연결된 차선에 대한 인덱스 추출  ex) [[12, 8], [12,7]]
        step2. 추출된 lane list에 대해서 filtering된 속도 및 가속도를 기반으로 일정 미래 시간까지 점유하는 포인트 추출
        """
        
        self.OccupiedPt = {}
        for agent_id in self.FilteredHistory.keys():
                    
            occupied_pt = []
            agent = self.FilteredHistory[agent_id]
            
            filtered_x = agent[-1][0]
            filtered_y = agent[-1][1]                        
            filtered_v = agent[-1][2]
            filtered_a = agent[-1][3]
            filtered_h = agent[-1][4]

            lane_cand = self.get_possible_lane(filtered_x, filtered_y, filtered_h)                             ### step 1 ###             
            self.PossiblePath[agent_id] = lane_cand
            
            for lane_list in lane_cand:                                                                        ### step 2 ### 
                
                cur_lane_pt = np.concatenate([self.map_pt[lane_id] for lane_id in lane_list], axis=0) 
                dist_to_lane = (cur_lane_pt[:,0]-filtered_x)**2 + (cur_lane_pt[:,1]-filtered_y)**2
                cur_lane_pt_idx = np.argmin(dist_to_lane)
                cur_lane_pt_idx = np.clip( cur_lane_pt_idx-3 , 0, 30) 
                cur_lane_pt = cur_lane_pt[cur_lane_pt_idx:]
                cur_lane_s = np.cumsum(np.linalg.norm(cur_lane_pt[1:,:2] - cur_lane_pt[:-1,:2], axis=-1), axis=-1)
                
                target_t = 2   # 얼마까지의 미래를 볼 것인지 설정 (~TTC)
                safety_margin = 10
                
                exp_dist = filtered_v*target_t + np.clip(1/2*filtered_a*target_t**2, -filtered_v*target_t, 1e2) + safety_margin
                
                end_idx = np.where(cur_lane_s >= exp_dist)[0]
                occupied_pt.append(cur_lane_pt[:end_idx[0],:] if len(end_idx)>0 else cur_lane_pt)
            
            self.OccupiedPt[agent_id] = occupied_pt
    
    def Collision_Check(self, local_lane_info, Ego):
        
        ###########################
        ##### Collision_Check #### 
        ###########################
        
        """
        step1. object에 의해서 일정 시간동안 점유될 것이라고 예상된 point를 "occupied_list"에 저장
        step2. "occupied_list"가 없으면 -1 반환
        step3. local_lane_info를 "occupied_list"와 같은 global 좌표계로 수정
        step4. global 좌표계로 수정된 lane_info의 point 중 "occupied_list"와 일정 거리 이하인, 가장 가까운 인덱스 반환
        step5. 일정 거리 이하인 인덱스가 없는 경우 -1 반환
        """


        occupied_list = []                                                                                       ### step 1 ###     

        for key in self.OccupiedPt.keys():
            for i in range(len(self.OccupiedPt[key])):
                occupied_list.append(self.OccupiedPt[key][i])

        if len(occupied_list) == 0:                                                                               ### step 2 ###   
            return -1

        occupied_list = np.concatenate(occupied_list, axis=0)

        local_lane_x = local_lane_info[:,0]
        local_lane_y = local_lane_info[:,1]

        global_lane_x = local_lane_x * np.cos(Ego.h) - local_lane_y * np.sin(Ego.h) + Ego.x
        global_lane_y = local_lane_x * np.sin(Ego.h) + local_lane_y * np.cos(Ego.h) + Ego.y

        global_lane = np.stack([global_lane_x, global_lane_y], axis=1)                                            ### step 3 ###   

        for idx, global_lane_pt in enumerate(global_lane):
            if np.min(np.linalg.norm(occupied_list[:,:2]-global_lane_pt[np.newaxis,:], axis=-1))<2:                ### step 4 ###   
                return idx

        return -1                                                                                                 ### step 5 ###   

    def run(self):
        
        for id_ in self.vehicles.keys():
            if id_ == 0:
                
                sensor_info = self.vehicles[0].get_measure(self.vehicles)
                local_lane_info = self.vehicles[0].get_local_path()
                
                ##############################
                ### agent history tracking ###
                ##############################
                                
                self.Tracking(sensor_info)
                            
                ##############################
                ##### Maenuver Prediction #### 
                ##############################
                
                self.Maneuver_pred()
                
                ###############################
                ### LongitudinalController ###
                ###############################
                                                
                target_v = 15     ## target_v 선정
                ax_tarv = 1.5 * (1 - (self.vehicles[0].v / target_v)**4) ## target_v 에 도달하기 위한 가속도 추출
                ax_curve = self.vehicles[0].get_a_curvature()  ## lane 정보를 기반으로 curve를 돌기 위한 가속도 추출
                
                
                ax_surv = 9.8  ## 주변 차량들이 점유한 point를 기반으로 주변 차량을 고려한 가속도 추출 
                idx = self.Collision_Check(local_lane_info, self.vehicles[0]) ## 주변 차량들이 점유한 포인트에 대해 자차량 path에서 가장 가까운 인덱스 추출 
                
                if idx >=0:
                    path_col = local_lane_info[:idx]
                    dist_to_col = np.sum(np.linalg.norm(path_col[1:,:2] - path_col[:-1,:2], axis=-1)) ## collsion point까지 거리 계산
                    ax_surv = self.vehicles[0].IDM([dist_to_col, 5, self.vehicles[0].v]) ## collsion point 전에 멈추기 위한 가속도 계산

                ax = np.min([ax_surv, ax_tarv, ax_curve]) ## target 속도, curve, 주변 차량을 고려한 가속도 중 최솟값 선정
                ax = np.clip(ax, -4, 1.5) ## 안정성을 위한 감가속 제한
                
 
                ###############################
                #### LateralController(PP) ####
                ###############################

                lookahead_dist = self.vehicles[0].v * 1.0
                target_idx = np.where(local_lane_info[:,0]>=lookahead_dist)[0]
                if len(target_idx)==0:
                    target_idx = len(local_lane_info)-1
                else:
                    target_idx = target_idx[0]

                target_x = local_lane_info[target_idx, 0]
                target_y = local_lane_info[target_idx, 1]
                        
                r = (target_x**2+target_y**2)/(2*target_y+1e-2)

                delta = np.arctan2(6/r, 1)
 
                
                self.vehicles[id_].step_manual(ax = ax, steer = delta)

                
            if id_  > 0 :
                self.vehicles[id_].step_auto(self.vehicles, self.int_pt_list[id_])

                # lane_idx = self.get_lane_id(self.vehicles[id_].x, self.vehicles[id_].y, self.vehicles[id_].h)
                # print(id_, lane_idx)
            

    def respawn(self):
        if len(self.vehicles)<self.min_num_agent:
            self.spawn_agent()
            
    
    
#     def find_reachable_lanes(connectivity_matrix, current_lanes):
#     ` n = len(connectivity_matrix)
#         reachable_lanes = set(current_lanes)  # 시작점으로 주어진 lanes를 먼저 추가

#         # 현재 후보의 각 lane에 대해
#         for lane_idx in current_lanes:
#             # 모든 가능한 다른 lanes를 체크
#             for i in range(n):
#                 # 만약 lane_idx에서 i로 이동 가능하면
#                 if connectivity_matrix[lane_idx][i] == 1:
#                     reachable_lanes.add(i)  # 연결된 lane을 결과 목록에 추가

#         return list(reachable_lanes)  # 집합을 리스트로 변환하여 반환
# `

    def get_possible_lane(self, x, y, h):
        
        dist = np.linalg.norm(self.map_pt_list[:,:2] - np.array([x,y])[np.newaxis,:], axis=-1)
        
        head_dist = self.map_pt_list[:,2] - h
        head_dist = np.arctan2(np.sin(head_dist), np.cos(head_dist))*180/np.pi
        
        idx_cand = np.where(dist + 0.1*np.abs(head_dist)<1.2)[0]
        
        lane_idx = [bisect.bisect(self.map_pt_idx, idx) for idx in idx_cand]
        
        lane_idx = np.unique(lane_idx)
        
        lane_cand = lane_idx ## Current lane 
        
        ReachableLanes = []
        
        for lane_idx in lane_cand:
            reachable = [] 
            
            next_cand = np.where(self.connectivity[lane_idx,:]==1)[0]

            for cand_idx in next_cand:
                reachable.append([lane_idx, cand_idx])
                
            ReachableLanes+=reachable
        
        return ReachableLanes
        # target_path = []
        
        # for lane_id in lane_idx:
        #     connect_lane = np.where(self.connectivity[lane_id,:] == 1)[0]
            
        #     if len(connect_lane)==0:
        #         lane_cand += [ lane_id ]
        #     lane_cand+=[[lane_id, clane] for clane in connect_lane]
        
        return lane_cand
            
                    
if __name__ == '__main__':

    try:
        f = Environments()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start node.')


       
    
    
    
        
    
    
