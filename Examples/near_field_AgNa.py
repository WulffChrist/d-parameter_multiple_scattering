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
lmax = 11
res = 200
Rs = [4, 12]
gap = 3

omega = 3.25

fd = ['Ag','Na']

d = Rs[0]+Rs[1]+gap
#d3 = np.sqrt(d**2-(d/2)**2)
pos = np.array([[-d/2,0,0],[d/2,0,0]]) # dimer
#pos = np.array([[0,0,-d/2],[0,0,d/2]]) # dimer perpendicular
#pos = np.array([[-d/2,0,0],[d/2,0,0],[0,0,d3]]) # trimer
#pos = np.array([[-3*d/2,0,0],[-d/2,0,0],[d/2,0,0],[3*d/2,0,0]]) # chain
N = np.size(pos[:,0])

#epsin1 = Drude(omega,epsinf,omega_p1,gamma1)
#epsin2 = Drude(omega,epsinf,omega_p2,gamma2)
omegadata, epsAgR, epsAgI = np.loadtxt(fr'T_matrix\AgPermittivity.txt', unpack=True)
_, epsNaR, epsNaI = np.loadtxt(fr'T_matrix\NaPermittivity.txt', unpack=True)
epsinAg = epsAgR + 1j * epsAgI
#epsinAg = epsinAg[::Nskip]
epsinNa = epsNaR + 1j * epsNaI
#epsinNa = epsinNa[::Nskip]

mie = Mie(omega,epsout,lmax,mode)

epsinAgint = mie.dinter(omegadata,epsinAg)
epsinNaint = mie.dinter(omegadata,epsinNa)
epsin = np.array([epsinAgint,epsinNaint])

'''
xmin = 0
xmax = 0
zmin = 0
zmax = 0

for i in range(N):
    if (pos[i,0] + 2*Rs[i]) >= xmax:
        xmax = pos[i,0] + Rs[1]+Rs[0]
    if (pos[i,0] - 2*Rs[i]) <= xmin:
        xmin = pos[i,0] - 2*Rs[i]
    if (pos[i,2] + 2*Rs[i]) >= zmax:
        zmax = pos[i,2] + 1.5*Rs[1]
    if (pos[i,2] - 2*Rs[i]) <= zmin:
        zmin = pos[i,2] - 1.5*Rs[1]
'''
xmin = 2*Rs[0]
xmax = Rs[0]+Rs[1]
zmin = 1.5*Rs[1]
zmax = 1.5*Rs[1]
x = np.linspace(xmin,xmax,res)
z = np.linspace(zmin,zmax,res)

Z,X = np.meshgrid(z,x, indexing="xy")
Y = np.zeros_like(X)

a_inc = mie.plane_wave(E0,N)

E = mie.Efields(X,Y,Z,a_inc,Rs,pos,epsin,fd)

E_abs = np.sqrt(np.sum(np.abs(E)**2, axis=0))

plt.figure()
plt.pcolor(X,Z,E_abs)
plt.set_cmap('turbo')
plt.colorbar()
colors = ['pink','green']
for i in range(N):
    circle1 = plt.Circle((pos[i,0], pos[i,2]), Rs[i], color=colors[i], fill=True,linewidth=1)
    plt.gca().add_patch(circle1)
plt.xlabel("z")
plt.ylabel("x")
plt.show()

np.savetxt(fr'T_matrix\nearfield_data\dimer_{mode}_kx_Ez_X_omega{omega}_R1{Rs[0]}R2{Rs[1]}nm_dg{gap}nm_l{lmax}_res{res}.txt', X, delimiter='\t', fmt='%.14g')
np.savetxt(fr'T_matrix\nearfield_data\dimer_{mode}_kx_Ez_Z_omega{omega}_R1{Rs[0]}R2{Rs[1]}nm_dg{gap}nm_l{lmax}_res{res}.txt', Z, delimiter='\t', fmt='%.14g')
np.savetxt(fr'T_matrix\nearfield_data\dimer_{mode}_kx_Ez_E_omega{omega}_R1{Rs[0]}R2{Rs[1]}nm_dg{gap}nm_l{lmax}_res{res}.txt', Z, delimiter='\t', fmt='%.14g')