import numpy as np

def get_frenet(x, y, mapx, mapy):
    next_wp = next_waypoint(x, y, mapx, mapy)
    prev_wp = next_wp - 1 if next_wp > 0 else len(mapx) - 1

    n_x = mapx[next_wp] - mapx[prev_wp]
    n_y = mapy[next_wp] - mapy[prev_wp]
    x_x = x - mapx[prev_wp]
    x_y = y - mapy[prev_wp]

    if n_x == 0 and n_y == 0:
        proj_x = x_x
        proj_y = x_y
        proj_norm = 0
    else:
        proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)
        proj_x = proj_norm * n_x
        proj_y = proj_norm * n_y

    frenet_d = get_dist(x_x, x_y, proj_x, proj_y)

    ego_vec = [x - mapx[prev_wp], y - mapy[prev_wp], 0]
    map_vec = [n_x, n_y, 0]
    d_cross = np.cross(ego_vec, map_vec)
    
    if d_cross[-1] > 0:
        frenet_d = -frenet_d

    frenet_s = 0
    for i in range(prev_wp):
        frenet_s += get_dist(mapx[i], mapy[i], mapx[i + 1], mapy[i + 1])

    frenet_s += get_dist(0, 0, proj_x, proj_y)

    return frenet_s, frenet_d

def get_cartesian(s, d, mapx, mapy, maps):
    prev_wp = 0

    while (prev_wp < len(maps) - 2) and (s > maps[prev_wp + 1]):
        prev_wp += 1

    next_wp = np.mod(prev_wp + 1, len(mapx))

    dx = (mapx[next_wp] - mapx[prev_wp])
    dy = (mapy[next_wp] - mapy[prev_wp]) 

    heading = np.arctan2(dy, dx)  # [rad]

    seg_s = s - maps[prev_wp]
    seg_x = mapx[prev_wp] + seg_s * np.cos(heading)
    seg_y = mapy[prev_wp] + seg_s * np.sin(heading)

    perp_heading = heading + np.pi / 2
    x = seg_x + d * np.cos(perp_heading)
    y = seg_y + d * np.sin(perp_heading)

    return x, y, heading

def next_waypoint(x, y, mapx, mapy):
    closest_wp = get_closest_waypoints(x, y, mapx, mapy)
    map_x = mapx[closest_wp]
    map_y = mapy[closest_wp]

    heading = np.arctan2((map_y - y), (map_x - x))
    angle = np.abs(np.arctan2(np.sin(heading), np.cos(heading)))

    if angle > np.pi / 4:
        next_wp = (closest_wp + 1) % len(mapx)
        dist_to_next_wp = get_dist(x, y, mapx[next_wp], mapy[next_wp])

        if dist_to_next_wp < 5:
            closest_wp = next_wp

    return closest_wp


def get_closest_waypoints(x, y, mapx, mapy):
    min_len = 1e10
    closest_wp = 0

    for i in range(len(mapx)):
        _mapx = mapx[i]
        _mapy = mapy[i]
        dist = get_dist(x, y, _mapx, _mapy)

        if dist < min_len:
            min_len = dist
            closest_wp = i

    return closest_wp

def get_dist(x, y, _x, _y):
    return np.sqrt((x - _x) ** 2 + (y - _y) ** 2)