import numpy as np

import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from T_matrix.Multiple_Scattering import *
from T_matrix.Drude import *

hc=197.46358707 #eV*nm

epsinf = 1
omega_p= 5.9 #eV
gamma = 0.1 #eV
epsout = 1
E0 = 1

N=2

R = 20
gap = 0.1
mode = 'C'
lmax = 3

print(4**(-1.5)*47.1552)

Nomega = 200
omega = np.linspace(0.0001*omega_p,omega_p*0.999,Nomega)
epsin = Drude(omega,epsinf,omega_p,gamma)

mie = Mie(omega,epsout,lmax,mode)

lol = mie.dinter()

plt.figure()
plt.plot(omega,np.imag(lol))
plt.show()