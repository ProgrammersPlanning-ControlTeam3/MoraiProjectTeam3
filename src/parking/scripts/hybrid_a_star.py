#!/usr/bin/env python3

import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time
from dubins import Dubins
from obstacle import Obstacle
from map import map


from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from lib.mgeo.class_defs import *

show_animation  = True
"""
###################차량 스펙##################
max wheel Angle : -35~ 35
Length : 5.205m
Width : 1.495m
Wheelbase : 3.16 m
"""
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.heading = 3.14
        #g = f + h
        self.f = 0
        self.g = 0
        self.h = 0

# Check if position of node is same( if distance < threshold, regard as same node)
def isSamePosition(node_1, node_2, epsilon_position=0.8):
    SamePosition = False
    firstNodeX = node_1.position[0]
    firstNodeY = node_1.position[1]
    secondNodeX = node_2.position [0]
    secondNodeY = node_2.position [1]
    if np.sqrt((firstNodeX-secondNodeX)**2+(firstNodeY-secondNodeY)**2) < epsilon_position :
        SamePosition =True
    else:
        SamePosition = False
    return SamePosition # True or False

def isSameYaw(node_1, node_2, epsilon_yaw=0.2):
    SameYaw = False
    firstNodeYaw = node_1.position[2]
    secondNodeYaw = node_2.position[2]
    if abs(firstNodeYaw -secondNodeYaw) < epsilon_yaw:
        SameYaw = True
    else:
        SameYaw = False
    return SameYaw# True or False

# Action set, Moving only forward direction
def get_action(R, Vx, delta_time_step):
    yaw_rate = Vx / R
    distance_travel = Vx * delta_time_step
    # yaw_rate, delta_time_step, cost
    # left, right, 0.5 left , 0.5 right , straight
    action_set = [[yaw_rate, delta_time_step, distance_travel],
                  [-yaw_rate, delta_time_step, distance_travel],
                  [yaw_rate/2, delta_time_step, distance_travel],
                  [-yaw_rate/2, delta_time_step, distance_travel],
                  [0.0, delta_time_step, distance_travel]]
    return action_set

# Vehicle movement
def vehicle_move(position_parent, yaw_rate, delta_time, Vx):
    x_parent = position_parent[0]
    y_parent  = position_parent[1]
    yaw_parent = position_parent[2]
    R = Vx * delta_time

    # if yaw_rate != 0 (left or right turn)
    if yaw_rate != 0 :
        x_child = x_parent + R * np.cos(yaw_parent + yaw_rate * delta_time)
        y_child = y_parent + R * np.sin(yaw_parent + yaw_rate * delta_time)
        yaw_child = yaw_parent + yaw_rate * delta_time
    # move straight
    else:
        x_child = x_parent + R * np.cos(yaw_parent)
        y_child = y_parent + R * np.sin(yaw_parent)
        yaw_child = yaw_parent
    # yaw processing
    if yaw_child > 2 * np.pi:
        yaw_child = yaw_child - 2 * np.pi
    if yaw_child < 0:
        yaw_child = yaw_child + 2 * np.pi

    # return position : [x, y, yaw]
    return [x_child, y_child, yaw_child]

# 원과 직선의 거리를 구하는 함수
def distanceBetweenLindAndCircle(line_slope, line_intercept,circle_center):
    ob_x, ob_y = circle_center

    # 수직 거리 계산
    vertical_distance = abs(line_slope * ob_x - ob_y + line_intercept) / math.sqrt(line_slope**2 + 1)

    return vertical_distance

# 점과 점을 잇는 1차함수 구하기
def calculate_line_equation(point1, point2):
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]

    m = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    b = y1 - m * x1 if m != float('inf') else x1  # y = mx + b

    return m, b

# Collision check : path overlaps with any of obstacle
def collision_check(position_parent, yaw_rate, delta_time_step, obstacle_list, Vx):
    # 이전 노드를 기반으로 움직임 계산
    position_child = vehicle_move(position_parent,yaw_rate,delta_time_step,Vx)

    px = position_parent[0]
    py = position_parent[1]
    cx = position_child[0]
    cy = position_child[1]

    # 부모 노드와 아들노드 사이를 잇는 직선
    # m ,b = calculate_line_equation(position_parent,position_child)
    # grad = (cy-py) / (cx-px)

    col = False
    for obstacle in obstacle_list:

        if obstacle.is_inside(cx,cy) :
            col = True
            return True

    return col

# Check if the node is in the searching space
def isNotInSearchingSpace(position_child, space):
    NotInSearchingSpace = False
    nx = position_child[0]
    ny = position_child[1]
    min_x, max_x, min_y, max_y = space
    if (nx <= min_x ) or (max_x <= nx ) or (ny <= min_y ) or (max_y <= ny ):
        NotInSearchingSpace = True
    return NotInSearchingSpace

def heuristic(cur_node, goal_node):
    dist = np.sqrt((cur_node.position[0] - goal_node.position[0])**2 + (cur_node.position[1]  - goal_node.position[1])**2)
    return dist

def a_star(start, goal, space, obstacle_list, R, Vx, delta_time_step, weight):

    start_node = Node(None, position = start)
    goal_node = Node(None, position = goal)
    cnt = 0
    open_list = []
    closed_list = []
    yaw_rate = Vx / R
    open_list.append(start_node)

    while open_list != [] :
        # Find node with lowest cost
        cur_node = open_list[0]
        cur_index = 0

        for index, node in enumerate(open_list):
            # print(node.position, node.f)
            if node.f < cur_node.f :
                cur_node = node
                cur_index = index

        # print("cur node = ", cur_index, cur_node.position, cur_node.f)

        # If goal, return optimal path
        if (isSamePosition(cur_node, goal_node, epsilon_position=0.6)):
            opt_path = []
            node = cur_node
            while node is not None :
                opt_path.append(node.position)
                node = node.parent
            return opt_path[::-1]

        # If not goal, move from open list to closed list
        open_list.pop(cur_index)
        closed_list.append(cur_node)

        # Generate child candidate
        action_set = get_action(R, Vx, delta_time_step)

        # action_set = [yaw_rate, delta_time, distance]
        #print("Let's get actions")
        for action in action_set:
            # R = action[2]
            yaw = action[0]
            # deltaX = R * np.cos(yaw)
            # deltaY = R * np.sin(yaw)

            child_candidate_position = vehicle_move(cur_node.position, yaw, delta_time_step ,Vx)
            #print(child_candidate_position)

            # If not in searching space, do nothing
            if isNotInSearchingSpace(child_candidate_position, space):
                #print("#Not In SearchingSpace!!")
                continue

            # If collision expected, do nothing
            #print(cur_node.position)
            if collision_check(cur_node.position, yaw, delta_time_step, obstacle_list, Vx) :
                #print("#Collision!!")
                continue

            # If not collision, create child node
            child = Node(cur_node, child_candidate_position)

            # If already in closed list, do nothing
            isInClosedList = False
            for CompareNode in closed_list :
                if isSamePosition(child, CompareNode) and isSameYaw(child, CompareNode) :
                    isInClosedList = True
                    break

            #do nothing
            if isInClosedList is True :
                #print("is Same node!")
                continue

            # If not in closed list, update open list
            child.g = cur_node.g + action[2]
            child.h = heuristic(child,goal_node)
            child.f = child.g + weight * child.h
            if len(open_list) !=0 :
                #find same node in open_list
                for CompareNode in open_list :
                    if isSamePosition(child, CompareNode) and isSameYaw(child, CompareNode) :
                        if child.f < CompareNode.f :
                            CompareNode.parent = child.parent
                            CompareNode.f = child.f
                else:
                    open_list.append(child)
            else:
                open_list.append(child)
        # show graph
        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.5)
            if len(closed_list) % 100 == 0:
                plt.pause(0.001)
        #time.sleep(0.1)
        # cnt +=1
        # if cnt == 10:
        #     break
    print("vehicle can't reach to goal")


def main():

    start, goal, obstacle_list, space = map()

    if show_animation == True:
        theta_plot = np.linspace(0,1,101) * np.pi * 2
        plt.figure(figsize=(8,8))
        plt.plot(start[0], start[1], 'bs',  markersize=7)
        plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs',  markersize=7)
        plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)

        for obs in obstacle_list :
            obs.plot()

        plt.axis(space)
        # plt.grid(True)
        plt.xlabel("X [m]"), plt.ylabel("Y [m]")
        # plt.title("Hybrid a star algorithm", fontsize=20)

    opt_path = a_star(start, goal, space, obstacle_list, R=4.51, Vx=4.0, delta_time_step=0.5, weight=1.1)

    print("Optimal path found!")
    len_opt_path = len(opt_path)

    #Dubins
    dubins = Dubins()
    kappa_ = .25/2.0
    dubins_base = [opt_path[0],opt_path[int(0.3*len_opt_path)],opt_path[int(0.6*len_opt_path)],opt_path[-1]]
    dubins_global_path = []
    
    #포인트 간의 dubins path 를 추출해줌
    for i in range(len(dubins_base)-1) :
        cartesian_path, _, _ = dubins.plan(dubins_base[i], dubins_base[i+1], kappa_)
        path_x, path_y, path_yaw = cartesian_path
        plt.plot(path_x, path_y, 'g-')
        
    print(path_x)


    opt_path = np.array(opt_path)
    if show_animation == True:
        plt.plot(opt_path[:,0], opt_path[:,1], "m.-")
        plt.show()

def get_hybrid_a_star_dubins_global_path():

    start, goal, obstacle_list, space = map()
    opt_path = a_star(start, goal, space, obstacle_list, R=4.51, Vx=4.0, delta_time_step=0.5, weight=1.1)
    print("Optimal path found!")
    len_opt_path = len(opt_path)

    #Dubins
    dubins = Dubins()
    kappa_ = .25/2.0
    dubins_base = [opt_path[0],opt_path[int(0.3*len_opt_path)],opt_path[int(0.6*len_opt_path)],opt_path[-1]]
    dubins_global_path = []
    dubins_x = []
    dubins_y = []

    #포인트 간의 dubins path 를 추출해줌
    #global path = x,y,
    for i in range(len(dubins_base)-1) :
        cartesian_path, _, _ = dubins.plan(dubins_base[i], dubins_base[i+1], kappa_)
        path_x, path_y, path_yaw = cartesian_path
        dubins_x += path_x
        dubins_y += path_y
    dubins_waypoint = zip(dubins_x,dubins_y)

    for waypoint in dubins_waypoint :
        path_x = waypoint[0]
        path_y = waypoint[1]
        read_pose = PoseStamped()
        read_pose.pose.position.x = path_x
        read_pose.pose.position.y = path_y
        read_pose.pose.orientation.w = 1
        dubins_global_path.poses.append(read_pose)
        
    return dubins_global_path

if __name__ == "__main__":
    main()



