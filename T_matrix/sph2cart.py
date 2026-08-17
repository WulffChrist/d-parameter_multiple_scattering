import numpy as np

def sph2cart(r,theta,phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z

def cart2sph(x,y,z):
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arctan2(np.sqrt(x**2+y**2), z)
    phi = np.arctan2(y,x)
    phi = np.mod(phi, 2*np.pi)
    """ if np.size(y) == 1:
        if y == 0:
            phi = 0
    else:
        mask = y == 0
        phi[mask] = 0 """
    return r,theta,phi

def E_sph2cart(E,theta,phi):
    Exyz = np.array([
        E[0]*np.sin(theta)*np.cos(phi) + E[1]*np.cos(theta)*np.cos(phi) - E[2]*np.sin(phi),
        E[0]*np.sin(theta)*np.sin(phi) + E[1]*np.cos(theta)*np.sin(phi) + E[2]*np.cos(phi),
        E[0]*np.cos(theta) - E[1]*np.sin(theta)
    ])
    return Exyz