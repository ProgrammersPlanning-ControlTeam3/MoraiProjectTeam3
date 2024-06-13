import numpy as np
class BicycleModel:
    def __init__ (self, model_name, wheelbase):
        self.model_name=model_name
        self.wheelbase = wheelbase

    def dynamic_function(self, x, u):
        '''
         x: (x position, y position heading angle theta, velocity)
         u : input vector(acceleration, steering angle)
        '''
        # Initialize the array to calculate the derivative of state vector
        dxdt= np .zeros_like(x)

        # Calculating the derivative...
        dxdt[0] = x[3] * np.cos(x[2]) # Position of X
        dxdt[1] = x[3] * np.sin(x[2]) # Position of Y
        dxdt[2] = x[3] * (np.tan(u[1]) /self.wheelbase) # Heading Angle
        dxdt[3] = u[0] # Velocity
        return dxdt

    def get_linearized_dynamics_continuous(self, x, u):
        # x : Current State Vector
        # U : Current Input Vector
        A = np.zeros((4,4)) # State Transition Vector
        G = np.zeros((4,2)) # Mapping Matrix
        A[0 , 2] = -x[3]*np.sin(x[2]) # ∂x'/∂θ
        A[0, 3] = np.cos(x[2])        # ∂x'/∂v
        A[1,2 ] = x[3]*np.cos(x[2])   # ∂y'/∂θ
        A[1,3] = np.sin(x[2])         # ∂y'/∂v
        A[2,3] = np.tan(u[1]) / self.wheelbase # ∂θ'/∂v

        G[2,1] = (x[3] / self.wheelbase) * (1/np.cos(u[1])**2) # ∂θ'/∂δ : The effects of the steering angle
        G[3,0] = 1  # ∂v'/∂a The effects of Acc.
        return A, G