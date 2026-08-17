import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from T_matrix.Multiple_Scattering import *
from T_matrix.Drude import *

hc = 197.46358707 #eV*nm

epsinf = 1
epsout = 1
E0 = 1

mode = 'Q'
lmax = 5
res = 200
R = 20
gap = 5

omega_p1 = 5.9 #eV
gamma1 = 0.1 #eV

omega_p2 = 5.9 #eV
gamma2 = 0.1 #eV

omega = 0.5*omega_p1

d = 2*R+gap
d3 = np.sqrt(d**2-(d/2)**2)
pos = np.array([[-d/2,0,0],[d/2,0,0]]) # dimer
#pos = np.array([[0,0,-d/2],[0,0,d/2]]) # dimer perpendicular
#pos = np.array([[-d/2,0,0],[d/2,0,0],[0,0,d3]]) # trimer
#pos = np.array([[-3*d/2,0,0],[-d/2,0,0],[d/2,0,0],[3*d/2,0,0]]) # chain
""" pos = np.array([[-d/2,0,0],[d/2,0,0],
                [0,0,d3],[-d,0,d3],[d,0,d3],
                [-d/2,0,2*d3],[d/2,0,2*d3],
                [0,0,-d3],[-d,0,-d3],[d,0,-d3],
                [0,0,3*d3],[-d,0,3*d3],[d,0,3*d3]
                ]) # hexagon with one in the middle """
N = np.size(pos[:,0])

Rs = np.ones_like(pos[:,0])
Rs = Rs*R

epsin1 = Drude(omega,epsinf,omega_p1,gamma1)
epsin2 = Drude(omega,epsinf,omega_p2,gamma2)

epsin = np.array([epsin1,epsin2])

mie = Mie(omega,epsout,lmax,mode)

xmin = 0
xmax = 0
zmin = 0
zmax = 0
for i in range(N):
    if (pos[i,0] + 2*Rs[i]) >= xmax:
        xmax = pos[i,0] + 2*Rs[i]
    if (pos[i,0] - 2*Rs[i]) <= xmin:
        xmin = pos[i,0] - 2*Rs[i]
    if (pos[i,2] + 2*Rs[i]) >= zmax:
        zmax = pos[i,2] + 2*Rs[i]
    if (pos[i,2] - 2*Rs[i]) <= zmin:
        zmin = pos[i,2] - 2*Rs[i]

x = np.linspace(xmin,xmax,res)
z = np.linspace(zmin,zmax,res)

Z,X = np.meshgrid(z,x, indexing="xy")
Y = np.zeros_like(X)

a_inc = mie.plane_wave(E0,N)

E = mie.Efields(X,Y,Z,a_inc,Rs,pos,epsin)

E_abs = np.sqrt(np.sum(np.abs(E)**2, axis=0))

plt.figure()
plt.pcolor(Z,X,E_abs)
plt.set_cmap('turbo')
plt.colorbar()
for i in range(N):
    circle1 = plt.Circle((pos[i,2], pos[i,0]), Rs[i], color='white', fill=False, linewidth=1)
    plt.gca().add_patch(circle1)
plt.xlabel("z")
plt.ylabel("x")
plt.show()