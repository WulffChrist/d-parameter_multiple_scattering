import numpy as np
import matplotlib.pyplot as plt
import time

import sys
from wakepy import keep
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from T_matrix.Multiple_Scattering import *
from T_matrix.Drude import *

start_time = time.time()

hc = 197.46358707 #eV*nm

epsinf = 1
epsout = 1
E0 = 1
lmaxs = [11]
modes = ['Q']

Nomega = 500
#Nskip = 10
Rs = [4, 12]
gap = 3

omega_p1 = 4 #eV
gamma1 = 0.1 #eV

omega_p2 = 4 #eV
gamma2 = 0.1 #eV

fd = ['Ag','Na']

#omega = np.linspace(0.25*omega_p1,omega_p1*0.75,Nomega)
omega = np.linspace(2.5,4.5,Nomega)

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

with keep.running():
    for lmax in lmaxs:
        for mode in modes:
            print(lmax)
            mie = Mie(omega,epsout,lmax,mode)

            epsinAgint = mie.dinter(omegadata,epsinAg)
            epsinNaint = mie.dinter(omegadata,epsinNa)
            epsin = np.array([epsinAgint,epsinNaint])
            
            a_inc = mie.plane_wave(E0,N)

            Cext = mie.C_ext(a_inc,Rs,pos,epsin,fd)
            #Cscat = mie.C_scat(a_inc,Rs,pos,epsin)
            #Cabs = Cext - Cscat

            fig,axs = plt.subplots()
            #axs.plot(omega,Cext) #normalize to sum of (np.pi*Rs**2) /(2*np.pi*R**2)
            #axs[1].plot(omega/omega_p1,Cscat/(2*np.pi*R**2)) #normalize to sum of (np.pi*Rs**2)
            #axs[2].plot(omega/omega_p1,Cabs/(2*np.pi*R**2)) #normalize to sum of (np.pi*Rs**2)

            #axs[0].set_yscale('log')
            #axs[1].set_yscale('log')
            #axs[2].set_yscale('log')

            axs.set_xlabel(r'$\omega/\omega_{p1}$')
            #axs[1].set_xlabel(r'$\omega/\omega_{p1}$')
            #axs[2].set_xlabel(r'$\omega/\omega_{p1}$')

            axs.set_ylabel(r'$\sigma_{ext}/A_{geo}$')
            #axs[1].set_ylabel(r'$\sigma_{scat}/A_{geo}$')
            #axs[2].set_ylabel(r'$\sigma_{abs}/A_{geo}$')

            matrix = np.column_stack((omega, Cext))
            if mode == 'C':
                np.savetxt(fr'T_matrix\data\dimer_{mode}_kx_Ez_Qs_R1{Rs[0]}R2{Rs[1]}nm_dg{gap}nm_l{lmax}.txt', matrix, delimiter='\t', fmt='%.14g')
            if mode == 'Q':
                np.savetxt(fr'T_matrix\data\dimer_{mode}_{fd[0]}{fd[1]}_kx_Ez_Qs_R1{Rs[0]}R2{Rs[1]}nm_dg{gap}nm_l{lmax}.txt', matrix, delimiter='\t', fmt='%.14g')

            plt.show()
        
print(time.time()-start_time)