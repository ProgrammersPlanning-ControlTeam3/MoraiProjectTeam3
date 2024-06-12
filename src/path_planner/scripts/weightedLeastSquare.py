import sympy as sp
import numpy as np

class WeightedLeastSquare:
    def __init__(self, degree=9, start_end_weight_multiplier=4):
        self.degree = degree
        self.start_end_weight_multiplier = start_end_weight_multiplier
        self.x_sym = sp.symbols('x')
        self.b = sp.symbols(f'b_0:{degree + 1}')
        self.polynomial_func = sum(self.b[i] * self.x_sym**(degree - i) for i in range(degree + 1))
        self.polynomial_derivative = sp.diff(self.polynomial_func, self.x_sym)

    def evaluate_derivative(self, x_values):
        # For Jacobian... Fitting Function
        f_prime = sp.lambdify(self.x_sym, self.polynomial_derivative, 'numpy')
        return f_prime(x_values)

    def fit_curve(self, points):
        # points는 [x, y, yaw] 형식의 리스트
        x_data = np.array([p[0] for p in points])
        y_data = np.array([p[1] for p in points])
        yaw_data = np.array([p[2] for p in points])

        # 각 점의 방향을 고려한 가중치 계산
        predicted_tangent = np.tan(yaw_data)
        actual_tangent = self.evaluate_derivative(x_data)
        weights = np.exp(-np.abs(predicted_tangent - actual_tangent))  # 방향 차이에 따른 가중치
        weights[0] *= self.start_end_weight_multiplier
        weights[-1] *= self.start_end_weight_multiplier
        #  Weight Function
        W = np.diag(weights)

        # Jacobian Matrix
        J_func = sp.lambdify(self.x_sym, self.polynomial_derivative, 'numpy')
        J_matrix = np.vstack([J_func(x) for x in x_data])

        A = J_matrix.T @ W @ J_matrix
        B = J_matrix.T @ W @ y_data
        coefficients = np.linalg.solve(A, B)

        return coefficients