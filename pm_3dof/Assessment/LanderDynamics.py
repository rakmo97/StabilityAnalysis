# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:42:22 2021

@author: omkar_mulekar
"""

import numpy as np
from scipy import integrate


def LanderEOM(t,x,policy):
    
    # States:
    #   x[0]: x
    #   x[1]: y
    #   x[2]: z
    #   x[3]: dx
    #   x[4]: dy
    #   x[5]: dz
    
    # Parameters
    
    g = 9.81
    g0 = 9.81
    Isp = 300
    m = 1 # [kg]



    u = policy.predict(x.reshape(1,-1))[0]
    Fx = np.clip(u[0],-20,20)
    Fy = np.clip(u[1],-20,20)
    Fz = np.clip(u[2],  0,20)
    

    
    dx    = x[3]
    dy    = x[4]
    dz    = x[5]
    ddx   = (1/m)*Fx
    ddy   = (1/m)*Fy
    ddz   = (1/m)*Fz - g
    dm    = - np.sqrt(Fx**2 + Fy**2 + Fz**2) / (Isp*g0)

    xdot = np.array([dx,dy,dz,ddx,ddy,ddz])
    
    
    return xdot


