#!/usr/bin/env python3
import numpy as np

class CA():
    def __init__(self, dt=0.1):

        self.dt = dt

    def step(self, x):

        dt = self.dt
        x_new = [x[0]+(x[2]+1/2*x[3]*dt)*np.cos(x[4])*dt,
                 x[1]+(x[2]+1/2*x[3]*dt)*np.sin(x[4])*dt,
                 x[2]+x[3]*dt,
                 x[3],
                 x[4]]

        return np.array(x_new)

    def H(self, x):

        return np.array([x[0], x[1], x[2], x[4]])

    def JA(self, x, dt=0.1):

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

class CTRA_model():
    def __init__(self):
        pass

    def F(self,x,dt=0.1):

        if np.abs(x[5])>0.1:
            x_new = [x[0]+x[2]/x[5]*(np.sin(x[4]+x[5]*dt)-
                                                     np.sin(x[4]))+
                      x[3]/(x[5]**2)*(np.cos(x[4]+x[5]*dt)+
                                                dt*x[5]*np.sin(x[4]+x[5]*dt)-
                                                np.cos(x[4])),
                      x[1]+x[2]/x[5]*(-np.cos(x[4]+x[5]*dt)+
                                                     np.cos(x[4]))+
                      x[3]/(x[5]**2)*(np.sin(x[4]+x[5]*dt)-
                                                dt*x[5]*np.cos(x[4]+x[5]*dt)-
                                                np.sin(x[4])),
                      x[2]+x[3]*dt,
                      x[3],
                      x[4]+x[5]*dt,
                      x[5]]

        else:
            x_new = [x[0]+x[2]*np.cos(x[4])*dt+1/2*x[3]*np.cos(x[4])*dt**2,
                      x[1]+x[2]*np.sin(x[4])*dt+1/2*x[3]*np.sin(x[4])*dt**2,
                      x[2]+x[3]*dt,
                      x[3],
                      x[4]+x[5]*dt,
                      x[5]]


        return np.array(x_new)

    def H(self,x):

        return np.array([x[0],x[1],x[2],x[4]])

    def JA(self,x,dt = 0.1):

        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]
        r = x[5]


        # upper
        if np.abs(r)>0.1:
            JA_ = [[1,0,(np.sin(yaw+r*dt)-np.sin(yaw))/r,(-np.cos(yaw)+np.cos(yaw+r*dt)+r*dt*np.sin(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.cos(yaw+r*dt)-a*np.sin(yaw+r*dt)-v*r*np.cos(yaw)+a*np.sin(yaw))/r**2,
                    -2/r**3*((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))+
                    ((v+a*dt)*np.sin(yaw+r*dt)+dt*(r*v+a*r*dt)*np.cos(yaw+r*dt)-dt*a*np.sin(yaw+r*dt)-v*np.sin(yaw))/r**2],
                    [0,1,(-np.cos(yaw+r*dt)+np.cos(yaw))/r,(-np.sin(yaw)+np.sin(yaw+r*dt)-r*dt*np.cos(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))/r**2,
                    -2/r**3*((-r*v-a*r*dt)*np.cos(yaw+r*dt)+a*np.sin(yaw+r*dt)+v*r*np.cos(yaw)-a*np.sin(yaw))+
                    ((-v-a*dt)*np.cos(yaw+r*dt)+dt*(r*v+a*r*dt)*np.sin(yaw+r*dt)+a*dt*np.cos(yaw+r*dt)+v*np.cos(yaw))/r**2],
                    [0,0,1,dt,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,dt],
                    [0,0,0,0,0,1]]
        else:
            JA_ = [[1, 0 , np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2,-(v+1/2*a*dt)*np.sin(yaw)*dt ,0],
                    [0, 1 , np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2, (v+1/2*a*dt)*np.cos(yaw)*dt,0],
                    [0,0,1,dt,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,dt],
                    [0,0,0,0,0,1]]

        return np.array(JA_)

    def JH(self,x, dt = 0.1):
        px = x[0]
        py = x[1]
        v = x[2]
        a = x[3]
        yaw = x[4]
        r = x[5]

        # upper
        if np.abs(r)>0.1:

            JH_ = [[1,0,(np.sin(yaw+r*dt)-np.sin(yaw))/r,(-np.cos(yaw)+np.cos(yaw+r*dt)+r*dt*np.sin(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.cos(yaw+r*dt)-a*np.sin(yaw+r*dt)-v*r*np.cos(yaw)+a*np.sin(yaw))/r**2,
                    -2/r**3*((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))+
                    ((v+a*dt)*np.sin(yaw+r*dt)+dt*(r*v+a*r*dt)*np.cos(yaw+r*dt)-dt*a*np.sin(yaw+r*dt)-v*np.sin(yaw))/r**2],
                    [0,1,(-np.cos(yaw+r*dt)+np.cos(yaw))/r,(-np.sin(yaw)+np.sin(yaw+r*dt)-r*dt*np.cos(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))/r**2,
                    -2/r**3*((-r*v-a*r*dt)*np.cos(yaw+r*dt)+a*np.sin(yaw+r*dt)+v*r*np.cos(yaw)-a*np.sin(yaw))+
                    ((-v-a*dt)*np.cos(yaw+r*dt)+dt*(r*v+a*r*dt)*np.sin(yaw+r*dt)+a*dt*np.cos(yaw+r*dt)+v*np.cos(yaw))/r**2],
                    [0,0,1,dt,0,0],
                    [0,0,0,0,1,dt]]

        else:
            JH_ = [[1, 0 , np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2,-(v+1/2*a*dt)*np.sin(yaw)*dt ,0],
                    [0, 1 , np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2, (v+1/2*a*dt)*np.cos(yaw)*dt,0],
                    [0,0,1,dt,0,0],
                    [0,0,0,0,1,dt]]

        return np.array(JH_)


class CTRV_model():
    def __init__(self):
        pass
        
    def F(self, x, dt=0.1):

        if np.abs(x[4])>0.1:
            x_new = [x[0]+x[2]/x[4]*(np.sin(x[3]+x[4]*dt)-
                                                     np.sin(x[3])),
                      x[1]+x[2]/x[4]*(-np.cos(x[3]+x[4]*dt)+
                                                     np.cos(x[3])),
                      x[2],
                      x[3]+x[4]*dt,
                      x[4]]

        else:
            x_new = [x[0]+x[2]*np.cos(x[3])*dt,
                      x[1]+x[2]*np.sin(x[3])*dt,
                      x[2],
                      x[3]+x[4]*dt,
                      x[4]]

        return np.array(x_new)

    def H(self,x):

        return np.array([x[0],x[1],x[2],x[3]])


    def JA(self,x,dt = 0.1):
        px = x[0]
        py = x[1]
        v = x[2]
        yaw = x[3]
        r = x[4]

        if np.abs(r)>0.1:
            # upper
            JA_ = [[1, 0, (np.sin(yaw+r*dt)-np.sin(yaw))/r, v/r*(np.cos(yaw+r*dt)-np.cos(yaw)), 
                     -v/(r**2)*(np.sin(yaw+r*dt)-np.sin(yaw))+v/r*(dt*np.cos(yaw+r*dt))],
                    [0, 1, (np.sin(yaw+r*dt)-np.sin(yaw))/r, v/r*(np.cos(yaw+r*dt)-np.cos(yaw)), 
                     -v/(r**2)*(np.sin(yaw+r*dt)-np.sin(yaw))+v/r*(dt*np.cos(yaw+r*dt))],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, dt],
                    [0, 0, 0, 0, 1]]
        else:
            JA_ = [[1, 0 , np.cos(yaw)*dt, -v*np.sin(yaw)*dt ,0],
                   [0, 1 , np.sin(yaw)*dt, v*np.cos(yaw)*dt,0],
                   [0,0,1,0,0],
                   [0,0,0,1,dt],
                   [0,0,0,0,1]]

        return np.array(JA_)

    def JH(self,x,dt = 0.1):
      

        JH_ = [[1,0,0,0,0],
               [0,1,0,0,0],
               [0,0,1,0,0],
               [0,0,0,1,0]]

        return np.array(JH_)
