#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import obstacle as obstacle
from scenarioLoader import ScenarioLoader
"""
시나리오 파일을 바탕으로 주차장 map의 정보를 받아 옵니다.
TO DO:
시나리오 파일을 통해 goal,start Data가 잘 받아지는 지 확인
(랜덤시나리오 제작 후에도 잘 동작하는지 확인하기)

"""
def pify(alpha):
    v = np.fmod(alpha, 2*np.pi)
    if (v < - np.pi):
        v += 2 * np.pi
    else:
        v -= 2 * np.pi
    return v

def degrees_to_radians(degrees):
    return degrees * (np.pi / 180.0)

def map():
    # Start, goal : [x, y, theta]
    # default value == hard cording
    # start = [5.5, 1068.8048, degrees_to_radians(-35.765)]
    # goal = [5.25, 1021.73, degrees_to_radians(-120.769)]

    # Searching space : [min_x, max_x, min_y, max_y]
    space = [-18.19, 43.6, 1004.84, 1076.51]

    # Obstacle : (x, y, radius)
    obstacleList =[]
    car_size = [4.96, 2.069] # 4.96 , 2.069
    obstacleId = {
        'car' : 40100019,
        'corn' : 40100003
    }

    #load Data (start, goal , obstacle) from Scenario file
    scenario = ScenarioLoader()
    start_data = scenario.getParkingLotStart()
    start_pos = start_data['pos']
    start_yaw = degrees_to_radians(float(start_data['rot']['yaw']))
    start = [start_pos['x'],start_pos['y'],start_yaw]
    
    goal_data = scenario.getParkingLotGoal()
    goal_pos = goal_data['pos']
    goal_yaw = degrees_to_radians(float(goal_data['rot']['yaw'])+180)
    goal = [goal_pos['x'],goal_pos['y'],goal_yaw]

    objectList = scenario.getObject()

    for object in objectList :
        DataId = object['DataID']
        DataPos = object['pos']
        DataYaw = pify(degrees_to_radians(float(object['rot']['yaw'])))

        DataX = DataPos['x']
        DataY = DataPos['y']
        collision_weight = 1.2
        if DataId == obstacleId['car'] :
            obs = obstacle.RectangleObstacle(DataX,DataY,car_size[0],car_size[1],DataYaw,collision_weight)
            obstacleList.append(obs)

        elif DataId == obstacleId['corn'] :
            obs = obstacle.Obstacle(DataX,DataY, 0.5 , 1.8)
            obstacleList.append(obs)


    return start, goal, obstacleList, space

if __name__ == "__main__":
    theta_plot = np.linspace(0,1,101) * np.pi * 2
    start, goal, obstacle_list, space = map()
    plt.figure(figsize=(8,8))
    plt.plot(start[0], start[1], 'bs',  markersize=7)
    plt.text(start[0], start[1]+0.5, 'start', fontsize=12)

    plt.plot(goal[0], goal[1], 'rs',  markersize=7)
    plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
    for obs in obstacle_list :
        obs.plot()
    plt.axis(space)
    # plt.grid(True)
    plt.show()