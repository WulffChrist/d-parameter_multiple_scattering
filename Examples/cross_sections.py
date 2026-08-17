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
Nomega = 100
R = 20
gap = 10

omega_p1 = 5.9 #eV
gamma1 = 0.1 #eV

omega_p2 = 5.9 #eV
gamma2 = 0.1 #eV

omega = np.linspace(0.1*omega_p1,omega_p1*0.9,Nomega)

d = 2*R+gap
d3 = np.sqrt(d**2-(d/2)**2)
pos = np.array([[-d/2,0,0],[d/2,0,0]]) # dimer
#pos = np.array([[0,0,-d/2],[0,0,d/2]]) # dimer perpendicular
#pos = np.array([[-d/2,0,0],[d/2,0,0],[0,0,d3]]) # trimer
#pos = np.array([[-3*d/2,0,0],[-d/2,0,0],[d/2,0,0],[3*d/2,0,0]]) # chain
N = np.size(pos[:,0])

Rs = np.ones_like(pos[:,0])
Rs = Rs*R

epsin1 = Drude(omega,epsinf,omega_p1,gamma1)
epsin2 = Drude(omega,epsinf,omega_p2,gamma2)

epsin = np.array([epsin1,epsin2])

mie = Mie(omega,epsout,lmax,mode)

a_inc = mie.plane_wave(E0,N)

Cext = mie.C_ext(a_inc,Rs,pos,epsin)
Cscat = mie.C_scat(a_inc,Rs,pos,epsin)
Cabs = Cext - Cscat

fig,axs = plt.subplots(1,3)
axs[0].plot(omega/omega_p1,Cext/(2*np.pi*R**2)) #normalize to sum of (np.pi*Rs**2)
axs[1].plot(omega/omega_p1,Cscat/(2*np.pi*R**2)) #normalize to sum of (np.pi*Rs**2)
axs[2].plot(omega/omega_p1,Cabs/(2*np.pi*R**2)) #normalize to sum of (np.pi*Rs**2)

axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[2].set_yscale('log')

axs[0].set_xlabel(r'$\omega/\omega_{p1}$')
axs[1].set_xlabel(r'$\omega/\omega_{p1}$')
axs[2].set_xlabel(r'$\omega/\omega_{p1}$')

axs[0].set_ylabel(r'$\sigma_{ext}/A_{geo}$')
axs[1].set_ylabel(r'$\sigma_{scat}/A_{geo}$')
axs[2].set_ylabel(r'$\sigma_{abs}/A_{geo}$')

plt.show()