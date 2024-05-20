#!/usr/bin/env python3

import numpy as np

from dubins import Dubins, pify

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

class Obstacle(object):
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def plot(self, color='k'):
        theta = np.linspace(0, np.pi*2, num=30)
        x = self.x + self.r * np.cos(theta)
        y = self.y + self.r * np.sin(theta)

        plt.plot(x, y, color=color)

    def is_inside(self, x, y):
        dist = np.hypot(x - self.x, y - self.y)
        if dist <= self.r:
            return True
        else:
            return False


class RRTStar(object):
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

    def sample_free(self, obstacles, space):
        min_x, max_x, min_y, max_y = space
        if np.random.rand() > self.config["goal_sample_rate"]:
            rand_x = np.random.uniform(min_x, max_x)
            rand_y = np.random.uniform(min_y, max_y)
            rand_yaw = np.random.uniform(0, 2*np.pi)
            return np.array([rand_x, rand_y, rand_yaw])

        else:
            return self.goal

    def get_nearest(self, rand_node):
        # node: np.array with 2 elements
        min_dist = 1e10
        nearest_node_id = None
        for v in self.G.nodes:
            node = self.G.nodes[v]
            dist = np.hypot(rand_node[0] - node['x'], rand_node[1] - node['y'])
            if dist < min_dist:
                nearest_node_id = v
                min_dist = dist

        return nearest_node_id

    def steer(self, node_from, node_to, u=None):
        dubins = Dubins()
        curvature = 1.0/2.0
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
        curvature = 1.0/2.0
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
    min_x, max_x = -20, 20
    min_y, max_y = -20, 20

    #plot 공간
    space = [min_x, max_x, min_y, max_y]

    #시작지점 랜덤 설정
    start_position = np.random.uniform(low=-20, high=-5, size=2)
    start_yaw = np.random.uniform(low=0, high=2*np.pi, size=1)[0]
    start = [start_position[0], start_position[1], start_yaw]

    #goal 포지션 랜덤
    goal_position = np.random.uniform(low=5, high=20, size=2)
    goal_yaw = np.random.uniform(low=0, high=2*np.pi, size=1)[0]
    goal = [goal_position[0], goal_position[1], goal_yaw]


    #랜덤 obstacle 원형 
    obstacles = []
    for i in range(25):
        x = np.random.uniform(low=min_x, high=max_x, size=1)
        y = np.random.uniform(low=min_y, high=max_y, size=1)
        r = np.random.uniform(low=1.0, high=5.0)
        obstacle = Obstacle(x, y, r)

        if not obstacle.is_inside(start[0], start[1]) and not obstacle.is_inside(goal[0], goal[1]):
            obstacles.append(obstacle)

    #그리기
    for obs in obstacles:
        obs.plot()

    #rrt star config
    config = {
        "eta": 10.0,
        "gamma_rrt_star": 10.0,
        "goal_sample_rate": 0.05,
    }

    rrt_star = RRTStar(start, goal, config)

    is_first_node = True
    goal_node_id = None

    dubins = Dubins()
    kappa = 1/2.0

    for i in range(500):
        rand_node_state = rrt_star.sample_free(obstacles, space)
        # plt.plot(rand_node[0], rand_node[1], '.')

        nearest_node_id = rrt_star.get_nearest(rand_node_state)
        nearest_node_state = rrt_star.get_node(nearest_node_id)
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

        print(path)

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
