#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import obstacle as obstacle
from scenarioLoader import ScenarioLoader

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
    start = [5.5, 1068.8048, degrees_to_radians(-35.765)]
    goal = [5.25, 1021.73, degrees_to_radians(-120.769)]

    # Searching space : [min_x, max_x, min_y, max_y]
    space = [-18.19, 43.6, 1004.84, 1076.51]

    # Obstacle : (x, y, radius)
    obstacleList =[]
    car_size = [4.96, 2.069] # 4.96 , 2.069
    obstacleId = {
        'car' : 40100019,
        'corn' : 40100003
    }

    scenario = ScenarioLoader()
    objectList = scenario.getObject()

    for object in objectList :
        DataId = object['DataID']
        DataPos = object['pos']
        DataYaw = pify(degrees_to_radians(float(object['rot']['yaw'])))

        DataX = DataPos['x']
        DataY = DataPos['y']
        collision_weight = 0.75
        if DataId == obstacleId['car'] :
            obs = obstacle.RectangleObstacle(DataX,DataY,car_size[0],car_size[1],DataYaw,collision_weight)
            obstacleList.append(obs)

        elif DataId == obstacleId['corn'] :
            obs = obstacle.Obstacle(DataX,DataY,1.)
            obstacleList.append(obs)
    # obstacleList = []

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