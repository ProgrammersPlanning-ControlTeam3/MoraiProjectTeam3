#!/usr/bin/env python3

import numpy as np

from dubins import Dubins
from scenarioLoader import ScenarioLoader

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import obstacle as obstacle
import rectangle as Rectangle

"""
1. 출발지점과 골 그리고 주차장 범위 정하기 (완)
2. 주차범위 시각화 하기
3. obstacle 불러오기 (완)
4.

###################차량 스펙##################
max wheel Angle : -35~ 35
Length : 5.205m
Width : 1.495m
Wheelbase : 3.16 m
"""

kappa =  1.5/2.0
#g90 최소회전반경 r = 4.51

def twopify(alpha):
    return alpha - np.pi * 2 * np.floor(alpha / (np.pi * 2))

def pify(alpha):
    v = np.fmod(alpha, 2*np.pi)
    if (v < - np.pi):
        v += 2 * np.pi
    else:
        v -= 2 * np.pi
    return v

def degrees_to_radians(degrees):
    return degrees * (np.pi / 180.0)

class RRTStar(object):
    #start, goal
    def __init__(self, start, goal, config):
        self.G = nx.DiGraph()

        node_attrb = {
            'cost': 0,
            'x': start[0],
            'y': start[1],
            'yaw': start[2]
        }

        self.G.add_nodes_from([
            (-1, node_attrb)
        ])

        self.start = start
        self.goal = goal
        self.config = config

    #랜덤 노드 생성
    def sample_free(self, obstacles, space):
        min_x, max_x, min_y, max_y = space

        #goal 과의 거리가 정해진 값보다 멀다면 랜덤 노드생성
        # np.random.rand() -> 0~1 사이의 값 생성
        if np.random.rand() > self.config["goal_sample_rate"]:
            rand_x = np.random.uniform(min_x, max_x)
            rand_y = np.random.uniform(min_y, max_y)
            rand_yaw = np.random.uniform(0, 2*np.pi)

            return np.array([rand_x, rand_y, rand_yaw])

        else:
            return self.goal

    #가까운 노드 구함.
    def get_nearest(self, rand_node):
        # node: np.array with 2 elements
        min_dist = 1e10
        nearest_node_id = None

        for v in self.G.nodes:
            node = self.G.nodes[v]
            #거리를 구하는 함수
            dist = np.hypot(rand_node[0] - node['x'], rand_node[1] - node['y'])
            if dist < min_dist:
                nearest_node_id = v
                min_dist = dist

        return nearest_node_id

    # node와 노드를 연결하고
    def steer(self, node_from, node_to, u=None):
        dubins = Dubins()
        curvature = kappa # 1.0 / 2.0

        #노드와 노드사이 길이 있는지 판별
        path, _, dubins_path = dubins.plan([node_from[0], node_from[1], node_from[2]],
                                           [node_to[0], node_to[1], node_to[2]], curvature)
        if path is None:
            return None

        path_length = dubins_path.length()
        path_x, path_y, path_yaw = path

        # node_attribute = {
        #     'x': node_to[0],
        #     'y': node_to[1],
        #     'yaw': node_to[2],
        #     'path_x': path_x,
        #     'path_y': path_y,
        #     'path_yaw': path_yaw,
        #     'path_length': path_length,
        #     'cost': 0.0
        # }

        return [node_to[0], node_to[1], node_to[2]]

    def get_node(self, node_id):
        node_state = np.array([self.G.nodes[node_id]['x'], self.G.nodes[node_id]['y'], self.G.nodes[node_id]['yaw']])
        return node_state

    def is_collision_free(self, node_from, node_to, obstacles):
        curvature = kappa
        path, _, dubins_path = dubins.plan([node_from[0], node_from[1], node_from[2]],
                                           [node_to[0], node_to[1], node_to[2]], curvature)

        path_x, path_y, path_yaw = path

        for x, y in zip(path_x, path_y):
            node_to_check = np.array([x, y])

            for i, obs in enumerate(obstacles):
                col = obs.is_inside(node_to_check[0], node_to_check[1])
                if col:
                    return False

        return True

    def get_near_node_ids(self, new_node, draw=False):
        card_v = len(list(self.G.nodes))
        radius = np.amin([
            self.config["gamma_rrt_star"] * np.sqrt(np.log(card_v) / card_v),
            self.config["eta"]
        ])

        if draw:
            theta = np.linspace(0, np.pi*2, num=30)
            x = new_node[0] + radius * np.cos(theta)
            y = new_node[1] + radius * np.sin(theta)

            plt.plot(x, y, 'g--', alpha=0.3)

        near_node_ids = []
        for v in self.G.nodes:
            node = self.G.nodes[v]
            dist = np.hypot(new_node[0] - node['x'], new_node[1] - node['y'])
            if dist < radius:
                near_node_ids.append(v)

        return near_node_ids

    def add_node(self, node_id, x, y, yaw):
        self.G.add_node(node_id, x=x, y=y, yaw=yaw)

    def get_node_cost(self, node_id):
        return self.G.nodes[node_id]['cost']

    def get_distance(self, node_from_id, node_to_id):
        node_from = self.G.nodes[node_from_id]
        node_to = self.G.nodes[node_to_id]

        dx = node_to['x'] - node_from['x']
        dy = node_to['y'] - node_from['y']
        return np.hypot(dx, dy)

    def add_edge(self, node_from_id, node_to_id, path):
        # print(" node_from_id: %d, node_to_id: %d " % (node_from_id, node_to_id))
        self.G.add_edge(node_from_id, node_to_id, path=path)

    def set_node_cost(self, node_id, cost):
        self.G.nodes[node_id]['cost'] = cost

    def get_parent(self, node_id):
        parents = list(self.G.predecessors(node_id))
        if len(parents) > 0:
            return parents[0]
        else:
            return None

    def remove_edge(self, node_from_id, node_to_id):
        self.G.remove_edge(node_from_id, node_to_id)

    def check_goal_by_id(self, node_id):
        node = self.G.nodes[node_id]

        dx = node['x'] - self.goal[0]
        dy = node['y'] - self.goal[1]
        dist = np.hypot(dx, dy)

        if dist < 1:
            return True
        else:
            return False


if __name__ == '__main__':
    #frame
    min_x, max_x = -21.75, 41.6
    min_y, max_y = 1005.84, 1076.51
    
    #parking lot space
    fig, ax = plt.subplots()
    p1,p2,p3,p4 = (-18.19,1031.92),(3.49,1074.75),(43.00,1053.83),(20.83,1004.84)
    parkingSpacePos = (p1,p2,p3,p4)
    parkingSpace = patches.Polygon(parkingSpacePos,facecolor="none",edgecolor="black",closed=True)
    ax.add_patch(parkingSpace)
    
    #object from Scenario
    scenario = ScenarioLoader()
    objectList = scenario.getObject()

    #plot 공간
    # space = [min_x, max_x, min_y, max_y]
    space = [-18.19, 43.00, 1004.84, 1074.75]
    parking_Rectangle = Rectangle.Rectangle(p1,p2,p3,p4)

    #시작지점 설정
    start_position = [5.5, 1068.8048]
    start_yaw = degrees_to_radians(-35.765) # 57.831
    start = [start_position[0], start_position[1], start_yaw]

    #goal 포지션
    goal_position = [5.25, 1021.73]
    goal_yaw = degrees_to_radians(-120.769) #-120.769
    goal = [goal_position[0], goal_position[1], goal_yaw]


    # obstacle 토픽 아니면 시나리오로 받기
    # 시나리오에서 장애물을 불러오면, 사이즈를 수기로 입력해야함.
    # DataID = 40100019 -> 주차된 차량 Rectangle Obstacle
    # 40100003 -> 콘 circle Obstacle
    obstacles = []
    car_size = [4.96, 2.069] # 4.96 , 2.069
    obstacleId = {
        'car' : 40100019,
        'corn' : 40100003
    }
    for object in objectList :
        DataId = object['DataID']
        DataPos = object['pos']
        DataYaw = pify(degrees_to_radians(float(object['rot']['yaw'])))

        DataX = DataPos['x']
        DataY = DataPos['y']

        if DataId == obstacleId['car'] :
            obs = obstacle.RectangleObstacle(DataX,DataY,car_size[0],car_size[1],DataYaw)
            obstacles.append(obs)
        elif DataId == obstacleId['corn'] :
            obs = obstacle.Obstacle(DataX,DataY,0.75)
            obstacles.append(obs)

    # obstacles = []

    #장애물 그리기
    for obs in obstacles:
        obs.plot()

    #rrt star config
    config = {
        "eta": 10.0,
        #gamma = 최적화 범위
        "gamma_rrt_star": 2.0,  # 2.0
        "goal_sample_rate": 0.1, # 0.05
    }

    rrt_star = RRTStar(start, goal, config)

    is_first_node = True
    goal_node_id = None

    dubins = Dubins()
    # 1.0/2.0

    for i in range(500):
        #make random node
        #랜덤으로 goal 과 같은 노드가 생성됨.
        rand_node_state = rrt_star.sample_free(obstacles, space)
        # plt.plot(rand_node_state[0], rand_node_state[1], '.')

        #while node is in parking lot
        while not parking_Rectangle.is_inside(rand_node_state) :
            rand_node_state = rrt_star.sample_free(obstacles, space)

        #draw node
        # plt.plot(rand_node_state[0], rand_node_state[1], '.')

        #랜덤으로 생성된 노드와 가장 가까운 노드
        nearest_node_id = rrt_star.get_nearest(rand_node_state)
        nearest_node_state = rrt_star.get_node(nearest_node_id)

        #
        new_node_state = rrt_star.steer(nearest_node_state, rand_node_state)

        if new_node_state is None:
            continue
        # plt.plot(new_node[0], new_node[1], 's')

        if rrt_star.is_collision_free(nearest_node_state, new_node_state, obstacles):
            near_node_ids = rrt_star.get_near_node_ids(new_node_state, draw=True)
            path, _, dubins_path = dubins.plan(nearest_node_state, new_node_state, kappa)
            if path is not None:
                rrt_star.add_node(i, x=new_node_state[0], y=new_node_state[1], yaw=new_node_state[2])
                if is_first_node:
                    rrt_star.add_edge(-1, i, path)
                    is_first_node = False
                plt.plot(new_node_state[0], new_node_state[1], 's')

                min_node_id = nearest_node_id
                # min_cost = rrt_star.get_node_cost(nearest_node_id) + rrt_star.get_distance(i, nearest_node_id)
                min_cost = rrt_star.get_node_cost(nearest_node_id) + dubins_path.length()
                min_path = path

                # Connect along a minimum-cost path
                for near_node_id in near_node_ids:
                    near_node_state = rrt_star.get_node(near_node_id)
                    if rrt_star.is_collision_free(near_node_state, new_node_state, obstacles):
                        path, _, dubins_path = dubins.plan(near_node_state, new_node_state, kappa)
                        if path is not None:
                            cost = rrt_star.get_node_cost(near_node_id) + dubins_path.length()
                            if cost < min_cost:
                                min_node_id = near_node_id
                                min_cost = cost
                                min_path = path

                if min_path is not None:
                    rrt_star.set_node_cost(i, min_cost)
                    rrt_star.add_edge(min_node_id, i, min_path)

                # Rewire the tree
                for near_node_id in near_node_ids:
                    near_node_state = rrt_star.get_node(near_node_id)
                    if rrt_star.is_collision_free(new_node_state, near_node_state, obstacles):
                        path, _, dubins_path = dubins.plan(new_node_state, near_node_state, kappa)
                        if path is not None:
                            cost = rrt_star.get_node_cost(i) + dubins_path.length()
                            if cost < rrt_star.get_node_cost(near_node_id):
                                parent_node_id = rrt_star.get_parent(near_node_id)
                                if parent_node_id is not None:
                                    rrt_star.remove_edge(parent_node_id, near_node_id)
                                    rrt_star.add_edge(i, near_node_id, path)

                if rrt_star.check_goal_by_id(i):
                    goal_node_id = i
                    break


    path_on_edge = {}
    for (u, v, path) in rrt_star.G.edges.data('path'):
        plt.plot(path[0], path[1], 'b-')
        path_on_edge[(u, v)] = path

    # print(path_on_edge.keys())

    for e in rrt_star.G.edges:
        v_from = rrt_star.G.nodes[e[0]]
        v_to = rrt_star.G.nodes[e[1]]
        # path = path_on_edge[(v_from, v_to)]

        # rrt_star.G.edges(data=True)
        # edge_attribute = rrt_star.G.get_edge_data(v_from, v_to, default=0)

        # plt.plot([v_from['x'], v_to['x']], [v_from['y'], v_to['y']], 'b-')
        # plt.plot(path[0], path[1], 'b-')
        plt.text(v_to['x'], v_to['y'], e[1])

    if goal_node_id is not None:
        path = nx.shortest_path(rrt_star.G, source=-1, target=goal_node_id)
        xs = []
        ys = []


        for node_idx in range(len(path)-1):
            node_id = path[node_idx+1]
            prev_node_id = path[node_idx]
            node = rrt_star.G.nodes[node_id]
            edge = path_on_edge[(prev_node_id, node_id)]
            plt.plot(edge[0], edge[1], 'r-', lw=2)
        # plt.plot(xs, ys, 'r-', lw=3)

    # edge = path_on_edge[(11, 47)]
    # plt.plot(edge[0], edge[1], 'k-', lw=3)
    # node = rrt_star.G.node[11]
    # plt.plot(node['x'], node['y'], 'ro')
    #
    # node = rrt_star.G.node[47]
    # plt.plot(node['x'], node['y'], 'ro')

    plt.plot(start[0], start[1], 'ro')
    plt.plot(goal[0], goal[1], 'bx')

    plt.axis("equal")
    plt.show()
