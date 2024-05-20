#!/usr/bin/env python3

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#random map
#from map_5 import map
#fixed map
from map_6 import map

#add_node
#add_edge
#remove_edge
np.random.seed(50)

class RRTStar(object):
    def __init__(self, start, goal, config):
        self.G = nx.DiGraph() # Directed Graph

        self.G.add_nodes_from([(-1, {'cost': 0, 'x': start[0], 'y': start[1]})])

        # Add Start node [(Node ID, {'Cost', 'x', 'y'})]
        self.start = start
        self.goal = goal
        self.config = config

        # options
        self.max_iter = 5000
        self.goal_sample_rate = 0.5

        # self.success_dist_thres = success_dist_thres
        self.collision_check_step = 0.2
        self.stepsize = 0.5

    def is_inside (self, obstacle, x, y):
        ob_x, ob_y, r = obstacle.x , obstacle.y , obstacle.r
        dx = ob_x - x
        dy = ob_y - y
        # Not In Obstacle
        #print(np.sqrt(dx * dx + dy * dy), r)
        if np.sqrt(dx * dx + dy * dy) > r :
            return False
        # In Obstacle
        else:
            return True

    # Random point generation
    ## CHECKED
    def sample_free(self, obstacles, space):
        min_x, max_x, min_y, max_y = space

        if np.random.rand() < 0.5: #self.goal_sample_rate:
            return self.goal
        while True:
            rand_x = np.random.uniform(min_x, max_x)
            rand_y = np.random.uniform(min_y, max_y)
            #print(rand_x,rand_y)
            isInObstacle = any(self.is_inside(obstacle, rand_x, rand_y) for obstacle in obstacles)
            # for ob in obstacles :
            #     is_inside_obs = self.is_inside(ob,rand_x,rand_y)
            #     #print(self.is_inside(ob,rand_x,rand_y))
            #     if not is_inside_obs :
            if not isInObstacle:
                return np.array([rand_x, rand_y])


    # Search nearest node
    ## CHECKED
    def get_nearest(self, rand_node):
        min_index = 0
        min_dist = 1e9
        #print(self.G.nodes)
        for i in range(0, len(self.G.nodes)) :
            #print(i)
            node_id= list(self.G.nodes)[i]
            #print(node_id)
            node = self.get_node(node_id)
            dx = rand_node[0] - node[0]
            dy = rand_node[1] - node[1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                min_dist = dist
                min_index = node_id
        # return index
        return min_index

    # Node connection
    # u -> random value
    ## CHECKED
    def steer(self, node_from, node_to, u=None):
        from_x = node_from[0]
        from_y = node_from[1]
        to_x = node_to[0]
        to_y = node_to[1]

        dx = to_x - from_x
        dy = to_y - from_y
        mag = np.sqrt(dx*dx + dy*dy)

        direction_x = dx/mag
        direction_y = dy/mag

        new_x = from_x + u * direction_x
        new_y = from_y + u * direction_y

        return new_x, new_y

    # Returns node(2d-array with position info.) corresponding to the node id
    ## CHECKED
    def get_node(self, node_id):
        node = np.array([self.G.nodes[node_id]['x'], self.G.nodes[node_id]['y']])
        return node

    # Collision Check
    ## CHECKED
    def is_collision_free(self, node_from, node_to, obstacles, step=0.2):
        dx = node_to[0] - node_from[0]
        dy = node_to[1] - node_from[1]

        length = np.sqrt(dx*dx + dy*dy)

        direction_x = dx/length
        direction_y = dy/length

        nodes_to_check = [node_from, node_to]

        if length >= step :
            n_add = int(np.floor(length/step))
            for i in range(n_add):
                step_node_x = node_from[0] + step * direction_x * (i+1.0)
                step_node_y = node_from[1] + step * direction_y * (i+1.0)
                nodes_to_check.append(np.array([step_node_x,step_node_y]))
        for node in nodes_to_check:
            for obstacle in obstacles:
                obs_x = obstacle.x
                obs_y = obstacle.y
                obs_r = obstacle.r
                dx = node[0] - obs_x
                dy = node[1] - obs_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist <= obs_r:
                    return False
        # Code
        # Collision check step size : u
        # Collision check : path(connection) between 2 nodes
        return True

    # Find adjacent nodes
    ##Checking
    def get_near_node_ids(self, new_node, draw):
        node_id = -1
        n = len(self.G.nodes)
        searching_distance = config['gamma_rrt_star'] * np.log(n) / n
        near_node_ids = []

        to_x = new_node[0]
        to_y = new_node[1]

        for i in range(0, len(self.G.nodes)) :
            node_id = list(self.G.nodes)[i]
            node = self.get_node(node_id)
            from_x = node[0]
            from_y = node[1]
            dx = to_x - from_x
            dy = to_y - from_y
            if np.sqrt(dx*dx + dy*dy) < searching_distance :
                near_node_ids.append(node_id)

        # Recommendation :
        # Searching distance proportional to log(# of nodes in tree)/(# of nodes in tree)
        return near_node_ids

    # Add node to tree
    ## CHECKED
    def add_node(self, node_id, x, y):
        self.G.add_node(node_id, x=x, y=y)

    # Get cost of designated node
    ## CHECKED
    def get_node_cost(self, node_id):
        return self.G.nodes[node_id]['cost']

    # Calculate the distance between 2 nodes
    def get_distance(self, node_from_id, node_to_id):
        from_node_x, from_node_y = self.get_node(node_from_id)
        to_node_x, to_node_y = self.get_node(node_to_id)
        dx = to_node_x - from_node_x
        dy = to_node_y - from_node_y
        dist = np.sqrt(dx*dx + dy*dy)
        return dist

    # Add edge(Connection) between 2 nodes
    def add_edge(self, node_from_id, node_to_id):
        self.G.add_edge(node_from_id, node_to_id)

    # Set cost to the node
    def set_node_cost(self, node_id, cost):
        self.G.nodes[node_id]['cost'] = cost

    # Get parent node of designated node : predecessors
    def get_parent(self, node_id):
        parents = list(self.G.predecessors(node_id))
        if len(parents) > 0:
            return parents[0]
        else:
            return None

    # Delete connections between 2 nodes
    def remove_edge(self, node_from_id, node_to_id):
        self.G.remove_edge(node_from_id, node_to_id)

    # Check goal
    def check_goal_by_id(self, node_id):
        node = self.G.nodes[node_id]
        dx = node['x'] - self.goal[0]
        dy = node['y'] - self.goal[1]
        dist = np.hypot(dx, dy)
        # Regard as destination, if distance between the node and the goal is smaller than threshold.
        if dist < self.config["goal_range"]:
            return True
        else:
            return False


if __name__ == '__main__':

    start, goal, space, obstacles = map(-20,20,-20,20)

    for obs in obstacles:
        obs.plot()

    config = {
        "eta": 3.0,
        "gamma_rrt_star": 4.0,
        "goal_sample_rate": 0.05,
        "min_u": 1.0,
        "max_u": 3.0,
        "goal_range" : 1.0
    }

    rrt_star = RRTStar(start, goal, config)

    is_first_node = True
    goal_node_id = None

    for i in range(500):

    ## STEP #1 Create Random Node
        #avoid obstacle
        #rand_node -> np.array()
        rand_node = rrt_star.sample_free(obstacles, space)
        plt.plot(rand_node[0], rand_node[1], '.')
        #print("rand_node", rand_node)

    ## STEP #2 Find Nearest Node
        nearest_node_id = rrt_star.get_nearest(rand_node)
        nearest_node = rrt_star.get_node(nearest_node_id)
        #print(nearest_node)

    ## STEP #3 Find Nearest Node
        new_node = rrt_star.steer(nearest_node, rand_node, np.random.uniform(config["min_u"],config["max_u"]))
        plt.plot(new_node[0], new_node[1], 's')

    ## STEP #4 Check Collision of the new node
        ###CHECKED Point###
        if rrt_star.is_collision_free(nearest_node, new_node, obstacles):
    ## Find adjacent nodes
            near_node_ids = rrt_star.get_near_node_ids(new_node, draw=True)

            rrt_star.add_node(i, new_node[0], new_node[1])
            if is_first_node:
                rrt_star.add_edge(-1, i)
                is_first_node = False

            plt.plot(new_node[0], new_node[1], 's')
            min_node_id = nearest_node_id
            min_cost = rrt_star.get_node_cost(nearest_node_id) + rrt_star.get_distance(i, nearest_node_id)

    ## Find a node with minimum cost among adjacent nodes
            for near_node_id in near_node_ids:
                near_node = rrt_star.get_node(near_node_id)
                if rrt_star.is_collision_free(near_node, new_node, obstacles):
                    cost = rrt_star.get_node_cost(near_node_id) + rrt_star.get_distance(near_node_id, i)
                    if cost < min_cost:
                        min_node_id = near_node_id
                        min_cost = cost

            rrt_star.set_node_cost(i, min_cost)
            rrt_star.add_edge(min_node_id, i)

    ## Rewire the tree with adjacent nodes
            for near_node_id in near_node_ids:
                near_node = rrt_star.get_node(near_node_id)
                if rrt_star.is_collision_free(new_node, near_node, obstacles):
                    cost = rrt_star.get_node_cost(i) + rrt_star.get_distance(i, near_node_id)
                    if cost < rrt_star.get_node_cost(near_node_id):
                        parent_node_id = rrt_star.get_parent(near_node_id)
                        if parent_node_id is not None:
                            rrt_star.remove_edge(parent_node_id, near_node_id)
                            rrt_star.add_edge(i, near_node_id)

    ## Check Goal
            if rrt_star.check_goal_by_id(i):
                goal_node_id = i
                break

    ## Plotting
    for e in rrt_star.G.edges:
        v_from = rrt_star.G.nodes[e[0]]
        v_to = rrt_star.G.nodes[e[1]]

        plt.plot([v_from['x'], v_to['x']], [v_from['y'], v_to['y']], 'b-')
        plt.text(v_to['x'], v_to['y'], e[1])

    if goal_node_id is not None:
        path = nx.shortest_path(rrt_star.G, source=-1, target=goal_node_id)
        xs = []
        ys = []

        for node_id in path:
            node = rrt_star.G.nodes[node_id]
            xs.append(node['x'])
            ys.append(node['y'])
        plt.plot(xs, ys, 'r-', lw=3)



    plt.plot(start[0], start[1], 'ro')
    plt.plot(goal[0], goal[1], 'bx')

    plt.axis("equal")
    plt.show()
