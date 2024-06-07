import math
import matplotlib.pyplot as plt

class SeparatingAxisTheorem:
    def __init__(self):
        pass

    @staticmethod
    def normalize(v):
        norm = math.sqrt(v[0] ** 2 + v[1] ** 2)
        return (v[0] / norm, v[1] / norm)

    @staticmethod
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    @staticmethod
    def edge_direction(p0, p1):
        return (p1[0] - p0[0], p1[1] - p0[1])

    @staticmethod
    def orthogonal(v):
        return (v[1], -v[0])

    @staticmethod
    def vertices_to_edges(vertices):
        return [SeparatingAxisTheorem.edge_direction(vertices[i], vertices[(i + 1) % len(vertices)]) \
                for i in range(len(vertices))]

    @staticmethod
    def project(vertices, axis):
        projections = [SeparatingAxisTheorem.dot(vertex, axis) for vertex in vertices]
        return (min(projections), max(projections))

    @staticmethod
    def contains(n, range_):
        a, b = range_
        if b < a:
            a, b = b, a
        return (n >= a) and (n <= b)

    @staticmethod
    def overlap(a, b):
        return SeparatingAxisTheorem.contains(a[0], b) or SeparatingAxisTheorem.contains(a[1], b) \
               or SeparatingAxisTheorem.contains(b[0], a) or SeparatingAxisTheorem.contains(b[1], a)

    def separating_axis_theorem(self, vertices_a, vertices_b):
        edges_a = self.vertices_to_edges(vertices_a)
        edges_b = self.vertices_to_edges(vertices_b)

        edges = edges_a + edges_b

        axes = [self.normalize(self.orthogonal(edge)) for edge in edges]

        for axis in axes:
            projection_a = self.project(vertices_a, axis)
            projection_b = self.project(vertices_b, axis)
            if not self.overlap(projection_a, projection_b):
                return False
        return True

    @staticmethod
    def get_vertice_rect(msg_tuple):
        center_x, center_y, yaw, L, W = msg_tuple
        vertex_3 = (center_x + (L / 2 * math.cos(yaw) - W / 2 * math.sin(yaw)),
                    center_y + (L / 2 * math.sin(yaw) + W / 2 * math.cos(yaw)))
        vertex_4 = (center_x + (-L / 2 * math.cos(yaw) - W / 2 * math.sin(yaw)),
                    center_y + (-L / 2 * math.sin(yaw) + W / 2 * math.cos(yaw)))
        vertex_1 = (center_x + (-L / 2 * math.cos(yaw) + W / 2 * math.sin(yaw)),
                    center_y + (-L / 2 * math.sin(yaw) - W / 2 * math.cos(yaw)))
        vertex_2 = (center_x + (L / 2 * math.cos(yaw) + W / 2 * math.sin(yaw)),
                    center_y + (L / 2 * math.sin(yaw) - W / 2 * math.cos(yaw)))
        return [vertex_1, vertex_2, vertex_3, vertex_4]
