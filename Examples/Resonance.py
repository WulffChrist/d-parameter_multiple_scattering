import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from T_matrix.Multiple_Scattering import *
from T_matrix.Drude import *
from T_matrix.single_sphere_resonance import *


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.figsize": (6.5,2.8),
    "xtick.top": False,
    "ytick.right": False,
    "axes.titlesize": 12,   # title font size
    "axes.labelsize": 12,   # x/y label font size
    "xtick.labelsize": 12,  # x tick labels
    "ytick.labelsize": 12,  # y tick labels
    "legend.fontsize": 12,  # legend text
    'font.size': 12,
    "figure.autolayout": True
})

hc=197.46358707 #eV*nm

epsinf = 1
omega_p= 5.9 #eV
gamma = 0.1 #eV
epsout = 1

R=np.linspace(1,50,100)
N=2
colors = ['tab:blue','tab:red','tab:green','tab:red']
#fig,axs = plt.subplots(2,1,sharex=True)
fig,axs = plt.subplots(1,2)
for i in range(N):
    n=i+1
    omega_lsp=omega_p/np.sqrt(1+(n+1)/n*epsout)
    omega_n=np.zeros([len(R),1],dtype=np.complex128)
    omega_nQ=np.zeros([len(R),1],dtype=np.complex128)
    for j in range(len(R)):
        omega_n[j] = resonance(n,R[j],epsinf,omega_p,gamma,epsout)
        omega_nQ[j] = resonanceQ(n,R[j],epsinf,omega_p,gamma,epsout)

    axs[1].semilogx(R,np.real(omega_n)/omega_p,label = fr'$l=${n}, C',color=colors[i])
    axs[1].semilogx(R,np.real(omega_nQ)/omega_p,label = fr'$l=${n}, Q',ls='dotted',color=colors[i])
    axs[1].axhline(omega_lsp/omega_p,ls='dashed',color=colors[i])
    #axs[1].semilogx(R,-np.imag(omega_n)/omega_p)
axs[1].set_ylabel(r'Re$[\omega_n/\omega_p]$')
#axs[1].set_ylabel(r'-Im$[\omega_n/\omega_p]$')

axs[1].legend(prop={'size': 10})
axs[1].set_xlim([1,50])
axs[1].invert_xaxis()
axs[1].set_xlabel('$R$ [nm]')
axs[1].text(1.7,0.47,r"\textbf{(b)}")
axs[1].set_xticks([1,2, 5, 10, 20, 50])
axs[1].xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.show()