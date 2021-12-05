import numpy as np
##POTENTIAL FLOWS
def uniform_stream(x,parameters):
    """Takes U and alpha as parameters - eg [2,np.pi/2]"""
    U = parameters[0]
    alpha = parameters[1]
    return U*np.cos(alpha)*x[0]+U*np.sin(alpha)*x[1]

def node_stream(x,parameters):
    """Takes m and (x_0,y_0) as parameters - eg [1,[1,1]]"""
    m = parameters[0]
    x_0,y_0 = parameters[1]
    return m*np.arctan2(x[1]-y_0,x[0]-x_0)/(2*np.pi)

def dipole_steam(x,parameters):
    mu = parameters[0]
    x_0,y_0 = parameters[1]
    alpha = parameters[2]
    r_2 = (x[0]-x_0)**2+(x[1]-y_0)**2
    return mu*(-(x[0]-x_0)*np.cos(alpha)+(x[1]-y_0)*np.sin(alpha))/(2*np.pi*r_2)

FLOW_DICT = {
        "uniform":uniform_stream,
        "node":node_stream,
        "dipole": dipole_steam
        }