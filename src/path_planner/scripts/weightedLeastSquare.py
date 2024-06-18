import sympy as sp
import numpy as np

class WeightedLeastSquare:
    def __init__(self, degree=9, start_end_weight_multiplier=4):
        self.degree = degree
        self.start_end_weight_multiplier = start_end_weight_multiplier
        self.x_sym = sp.symbols('x')
        self.b = sp.symbols(f'b_0:{degree+1}')
        self.polynomial_func = sum(self.b[i] *self.x_sym**(degree-i) for i in range(degree +1)) # N차 다항함수 만들기

    def evaluate(self, x_values):
        # Substitute x_values into the polynomial and evaluate
        return np.array([self.polynomial_func.subs(self.x_sym, x).evalf() for x in x_values])

    def fit_curve(self, points):
        # points는 [x, y, yaw] 형식의 리스트
        x_data = np.array([p[0] for p in points])
        y_data = np.array([p[1] for p in points])

        model_data = self.evaluate(x_data)
        residual = y_data - model_data
        J_matrix = np.array([[sp.diff(self.polynomial_func, b).subs(self.x_sym, x).evalf() for b in self.b] for x in x_data]) # 잔차에 대한 계수 편미분값
        J_matrix = np.array(J_matrix, dtype = np.float64)
        # weights = np.ones_like(x_data)
        weights = [100, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100]

        weights[0] *= self.start_end_weight_multiplier
        weights[-1] *= self.start_end_weight_multiplier
        W = np.diag(weights)
        A = J_matrix.T @ W @ J_matrix
        B = J_matrix.T @ W @ y_data
        A = np.array(A, dtype=np.float64)
        B = np.array(B, dtype=np.float64)

        try:
            # Try solving the system
            coefficients = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            # If A is singular, use the least squares method
            coefficients, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

        return coefficients

    def generate_local_path(self, x, y):
        local_path_points = []
        mapx = [pose.pose.position.x for pose in self.global_path_msg.poses]
        mapy = [pose.pose.position.y for pose in self.global_path_msg.poses]
        yaw = [pose.pose.orientation.w for pose in self.global_path_msg.poses]
        maps = [0]
        for i in range(1, len(mapx)):
            maps.append(maps[-1] + get_dist(mapx[i - 1], mapy[i - 1], mapx[i], mapy[i]))
        s, d = get_frenet(x, y, mapx, mapy)
        s_target = s + min(self.local_path_size, maps[-1] - s)
        s_targets = []
        d_targets = []
        yaw_targets = []
        for i in range(len(maps)):
            if maps[i] >= s_target:
                start_index = i
                break
        for i in range(start_index, min(start_index+10, len(maps))):
            s_point, d_point = get_frenet(mapx[i], mapy[i], mapx, mapy)
            yaw_frenet = self.yaw_to_frenet(yaw[i], mapx, mapy, i)
            s_targets.append(s_point)
            d_targets.append(d_point)
            yaw_targets.append(yaw_frenet)
        sd_pairs = list(zip(s_targets, d_targets, yaw_targets))
        T = 1.0
        path_function = WeightedLeastSquare()
        coefficients = path_function.fit_curve(sd_pairs)
        for s_val in np.linspace(s, s+self.local_path_size, num=self.local_path_size):
            d_val = self.evaluate_polynomial(coefficients, s_val)
            point_x, point_y, _ = get_cartesian(s_val, d_val, mapx, mapy, maps)
            local_path_points.append((point_x, point_y))
        return local_path_points

    def yaw_to_frenet(self, yaw, mapx, mapy, index):
        dx = mapx[index + 1] - mapx[index] if index < len(mapx) - 1 else mapx[index] - mapx[index - 1]
        dy = mapy[index + 1] - mapy[index] if index < len(mapy) - 1 else mapy[index] - mapy[index - 1]
        path_angle = np.atan2(dy, dx)
        frenet_yaw = yaw - path_angle  # 경로 탄젠트 각도와 yaw 각도의 차이 계산
        return frenet_yaw

    def evaluate_polynomial(self, coefficients, x):
        return sum(c * x ** i for i, c in enumerate(reversed(coefficients)))
