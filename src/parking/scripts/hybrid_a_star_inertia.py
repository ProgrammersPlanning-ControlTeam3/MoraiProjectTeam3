#!/usr/bin/env python3

import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time
import sys
import os

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)

from dubins import Dubins
from obstacle import Obstacle
from map import map


from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
# from lib.mgeo.class_defs import *

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
        self.direction = 2
        self.f = 0
        self.g = 0
        self.h = 0

def slice_list_with_interval(lst, interval):
    # 첫 번째 원소는 항상 포함
    result = [lst[0]]
    # 간격에 맞게 중간 원소들을 추가
    result.extend(lst[i] for i in range(interval, len(lst) - 1, interval))
    # 마지막 원소는 항상 포함
    if len(lst) > 1:  # 리스트가 1개 이상인 경우에만 마지막 원소를 추가
        result.append(lst[-1])
    return result

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
    # TODO: set action set L S R
    action_set = [[yaw_rate, delta_time_step, distance_travel],
                  [-yaw_rate, delta_time_step, distance_travel],
                  [yaw_rate/2, delta_time_step, distance_travel],
                  [-yaw_rate/2, delta_time_step, distance_travel],
                  [0.0, delta_time_step, distance_travel]]
    #LSR
    action_set = [[yaw_rate, delta_time_step, distance_travel],
                  [-yaw_rate, delta_time_step, distance_travel],
                #   [yaw_rate/2, delta_time_step, distance_travel],
                #   [-yaw_rate/2, delta_time_step, distance_travel],
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
        if (isSamePosition(cur_node, goal_node, epsilon_position=8)):
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
        for direction, action in enumerate(action_set):
            # R = action[2]
            yaw = action[0]
            # 0 right, 1 left , 2 straight
            direction = direction
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
            child.direction = direction
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

            direction_weight = 0.5
            if cur_node.direction == direction :
                child.f += direction_weight

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

    print("vehicle can't reach to goal")

## main = plt simulation
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
    print(opt_path)
    len_opt_path = len(opt_path)

    #Dubins
    dubins = Dubins()
    #kappa_ 가 높을수록 회전 반지름이 줄어듬.
    kappa_ = 1.5/2.0
    
    # dubins_base = [opt_path[0],opt_path[int(0.3*len_opt_path)],opt_path[int(0.6*len_opt_path)],opt_path[-1]]
    # dubins_base = [opt_path[0],opt_path[8],opt_path[-8],opt_path[-1]]
    # dubins_base = slice_list_with_interval(opt_path,5)
    
    dubins_base = [opt_path[-1],goal]
    dubins_global_path = []

    #포인트 간의 dubins path 를 추출해줌
    for i in range(1,len(dubins_base)-1) :
        cartesian_path, _, _ = dubins.plan(dubins_base[i], dubins_base[i+1], kappa_)
        path_x, path_y, path_yaw = cartesian_path
        plt.plot(path_x, path_y, 'g-')

    opt_path = np.array(opt_path)
    if show_animation == True:
        plt.plot(opt_path[:,0], opt_path[:,1], "m.-")
        plt.show()

    return dubins_global_path


def hybrid_a_star():
    #goal= [x,y ,yaw]
    start, goal, obstacle_list, space = map()
    opt_path = a_star(start, goal, space, obstacle_list, R=4.51, Vx=4.0, delta_time_step=0.5, weight=1.1)
    opt_path.append(goal)

    out_path = Path()

    #마지막 point 에서 주차goal 까지 dubins로 경로 생성.
    dubins = Dubins()
    kappa_ = .5/2.0
    cartesian_path, _,_ = dubins.plan(opt_path[-1],goal,kappa_)
    path_x , path_y , path_yaw = cartesian_path
    dubins_path = []

    # for i in range(len(path_x)) :
    #     opt_path.append([path_x[i],path_y[i]],path_yaw[i])
        # dubins_path.append([path_x[i],path_y[i]])
    # print(dubins_path)

    return opt_path, out_path, dubins_path

if __name__ == "__main__":
    #hybrid_a_star()
    main()


