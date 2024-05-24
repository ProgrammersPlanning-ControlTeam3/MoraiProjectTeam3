# coding: utf-8

import numpy as np

class CA():
    def __init__(self, dt=0.1):
        """
        CA 모델 클래스의 초기화 메서드

        Args:
            dt (float): 시간 간격 (기본값: 0.1)
        """
        self.dt = dt

    def step(self, x):
        """
        CA 모델의 한 단계 예측 수행

        Args:
            x (np.ndarray): 현재 상태 벡터 [x, y, v, a, theta]

        Returns:
            np.ndarray: 다음 시간 단계의 상태 벡터
        """
        dt = self.dt
        x_new = [x[0]+(x[2]+1/2*x[3]*dt)*np.cos(x[4])*dt,
                 x[1]+(x[2]+1/2*x[3]*dt)*np.sin(x[4])*dt,
                 x[2]+x[3]*dt,
                 x[3],
                 x[4]]

        return np.array(x_new)

    def H(self, x):
        """
        관측 모델

        Args:
            x (np.ndarray): 상태 벡터 [x, y, v, a, theta]

        Returns:
            np.ndarray: 관측 벡터
        """
        return np.array([x[0], x[1], x[2], x[4]])

    def JA(self, x, dt=0.1):
        """
        예측 모델의 야코비안

        Args:
            x (np.ndarray): 상태 벡터 [x, y, v, a, theta]
            dt (float): 시간 간격

        Returns:
            np.ndarray: 야코비안 행렬
        """
        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]

        JA_ = [[1, 0, np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2, -(v+1/2*a*dt)*np.sin(yaw)*dt],
               [0, 1, np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2,
                (v+1/2*a*dt)*np.cos(yaw)*dt],
               [0, 0, 1, dt, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]]

        return np.array(JA_)

    def JH(self, x, dt=0.1):
        """
        관측 모델의 야코비안

        Args:
            x (np.ndarray): 상태 벡터 [x, y, v, a, theta]
            dt (float): 시간 간격

        Returns:
            np.ndarray: 야코비안 행렬
        """
        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]

        JH_ = np.array([[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1]])

        return JH_


class CTRA():
    def __init__(self, dt=0.1):
        """
        CTRA 모델 클래스의 초기화 메서드

        Args:
            dt (float): 시간 간격 (기본값: 0.1)
        """
        self.dt = dt

    def step(self, x):
        """
        CTRA 모델의 한 단계 예측 수행

        Args:
            x (np.ndarray): 현재 상태 벡터 [x, y, v, a, theta, theta_rate]

        Returns:
            np.ndarray: 다음 시간 단계의 상태 벡터
        """
        if np.abs(x[5]) > 0.1:
            x_new = [x[0]+x[2]/x[5]*(np.sin(x[4]+x[5]*self.dt) -
                                     np.sin(x[4])) +
                     x[2]/(x[5]**2)*(np.cos(x[4]+x[5]*self.dt) +
                                     self.dt*x[5]*np.sin(x[4]+x[5]*self.dt) -
                                     np.cos(x[4])),
                     x[1]+x[2]/x[5]*(-np.cos(x[4]+x[5]*self.dt) +
                                     np.cos(x[4])) +
                     x[2]/(x[5]**2)*(np.sin(x[4]+x[5]*self.dt) -
                                     self.dt*x[5]*np.cos(x[4]+x[5]*self.dt) -
                                     np.sin(x[4])),
                     x[2]+x[3]*self.dt,
                     x[3],
                     x[4]+x[5]*self.dt,
                     x[5]]

        else:
            x_new = [x[0]+x[2]*np.cos(x[4])*self.dt,
                     x[1]+x[2]*np.sin(x[4])*self.dt,
                     x[2]+x[3]*self.dt,
                     x[3],
                     x[4],
                     x[5]]

        return np.array(x_new)

    def H(self, x):
        """
        관측 모델

        Args:
            x (np.ndarray): 상태 벡터 [x, y, v, a, theta, theta_rate]

        Returns:
            np.ndarray: 관측 벡터
        """
        return np.array([x[0], x[1], x[2], x[4]])

    def JA(self, x, dt=0.1):
        """
        예측 모델의 야코비안

        Args:
            x (np.ndarray): 상태 벡터 [x, y, v, a, theta, theta_rate]
            dt (float): 시간 간격

        Returns:
            np.ndarray: 야코비안 행렬
        """
        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]
        r = x[5]

        # upper
        if np.abs(r) > 0.1:
            JA_ = [[1, 0, (np.sin(yaw+r*dt)-np.sin(yaw))/r, (-np.cos(yaw)+np.cos(yaw+r*dt)+r*dt*np.sin(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.cos(yaw+r*dt)-a*np.sin(yaw+r*dt) -
                     v*r*np.cos(yaw)+a*np.sin(yaw))/r**2,
                    -2/r**3*((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw)) +
                    ((v+a*dt)*np.sin(yaw+r*dt)+dt*(r*v+a*r*dt)*np.cos(yaw+r*dt)-dt*a*np.sin(yaw+r*dt)-v*np.sin(yaw))/r**2],
                   [0, 1, (-np.cos(yaw+r*dt)+np.cos(yaw))/r, (-np.sin(yaw)+np.sin(yaw+r*dt)-r*dt*np.cos(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt) -
                     v*r*np.sin(yaw)-a*np.cos(yaw))/r**2,
                    -2/r**3*((-r*v-a*r*dt)*np.cos(yaw+r*dt)+a*np.sin(yaw+r*dt)+v*r*np.cos(yaw)-a*np.sin(yaw)) +
                    ((-v-a*dt)*np.cos(yaw+r*dt)+dt*(r*v+a*r*dt)*np.sin(yaw+r*dt)+a*dt*np.cos(yaw+r*dt)+v*np.cos(yaw))/r**2],
                   [0, 0, 1, dt, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, dt],
                   [0, 0, 0, 0, 0, 1]]
        else:
            JA_ = [[1, 0, np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2, -(v+1/2*a*dt)*np.sin(yaw)*dt, 0],
                   [0, 1, np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2,
                    (v+1/2*a*dt)*np.cos(yaw)*dt, 0],
                   [0, 0, 1, dt, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, dt],
                   [0, 0, 0, 0, 0, 1]]

        return np.array(JA_)

    def JH(self, x, dt=0.1):
        """
        관측 모델의 야코비안

        Args:
            x (np.ndarray): 상태 벡터 [x, y, v, a, theta, theta_rate]
            dt (float): 시간 간격

        Returns:
            np.ndarray: 야코비안 행렬
        """
        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]
        r = x[5]

        # upper
        if np.abs(r) > 0.1:

            JH_ = [[1, 0, (np.sin(yaw+r*dt)-np.sin(yaw))/r, (-np.cos(yaw)+np.cos(yaw+r*dt)+r*dt*np.sin(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.cos(yaw+r*dt)-a*np.sin(yaw+r*dt) -
                     v*r*np.cos(yaw)+a*np.sin(yaw))/r**2,
                    -2/r**3*((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw)) +
                    ((v+a*dt)*np.sin(yaw+r*dt)+dt*(r*v+a*r*dt)*np.cos(yaw+r*dt)-dt*a*np.sin(yaw+r*dt)-v*np.sin(yaw))/r**2],
                   [0, 1, (-np.cos(yaw+r*dt)+np.cos(yaw))/r, (-np.sin(yaw)+np.sin(yaw+r*dt)-r*dt*np.cos(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt) -
                     v*r*np.sin(yaw)-a*np.cos(yaw))/r**2,
                    -2/r**3*((-r*v-a*r*dt)*np.cos(yaw+r*dt)+a*np.sin(yaw+r*dt)+v*r*np.cos(yaw)-a*np.sin(yaw)) +
                    ((-v-a*dt)*np.cos(yaw+r*dt)+dt*(r*v+a*r*dt)*np.sin(yaw+r*dt)+a*dt*np.cos(yaw+r*dt)+v*np.cos(yaw))/r**2],
                   [0, 0, 1, dt, 0, 0],
                   [0, 0, 0, 0, 1, dt]]

        else:
            JH_ = [[1, 0, np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2, -(v+1/2*a*dt)*np.sin(yaw)*dt, 0],
                   [0, 1, np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2,
                    (v+1/2*a*dt)*np.cos(yaw)*dt, 0],
                   [0, 0, 1, dt, 0, 0],
                   [0, 0, 0, 0, 1, dt]]

        return np.array(JH_)

    def pred(self, x, t_pred):
        """
        주어진 시간만큼 예측하여 상태 벡터 목록 반환

        Args:
            x (np.ndarray): 초기 상태 벡터 [x, y, v, a, theta, theta_rate]
            t_pred (float): 예측할 시간

        Returns:
            np.ndarray: 예측된 상태 벡터들로 이루어진 목록
        """
        self.x = x

        x_list = [self.x]
        for t in range(int(t_pred/self.dt)):
            x_list.append(self.step(self.x))

        return np.array(x_list)
