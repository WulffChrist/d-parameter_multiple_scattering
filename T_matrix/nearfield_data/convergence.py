import numpy as np
import matplotlib.pyplot as plt

lmax=11
mode = 'Q_dmdp'
        
E10 = np.loadtxt(fr'T_matrix\nearfield_data\dimer_Q_kx_Ez_E_omega3.25_R14R212nm_dg3nm_l10_res40.txt', unpack=True)
X10 = np.loadtxt(fr'T_matrix\nearfield_data\dimer_Q_kx_Ez_X_omega3.25_R14R212nm_dg3nm_l10_res40.txt', unpack=True)
Z10 = np.loadtxt(fr'T_matrix\nearfield_data\dimer_Q_kx_Ez_Z_omega3.25_R14R212nm_dg3nm_l10_res40.txt', unpack=True)

E11 = np.loadtxt(fr'T_matrix\nearfield_data\dimer_Q_kx_Ez_E_omega3.25_R14R212nm_dg3nm_l11_res40.txt', unpack=True)
X11 = np.loadtxt(fr'T_matrix\nearfield_data\dimer_Q_kx_Ez_X_omega3.25_R14R212nm_dg3nm_l11_res40.txt', unpack=True)
Z11 = np.loadtxt(fr'T_matrix\nearfield_data\dimer_Q_kx_Ez_Z_omega3.25_R14R212nm_dg3nm_l11_res40.txt', unpack=True)

dx = X10[0,1]-X10[0,0]
dx11 = X11[0,1]-X11[0,0]
dz = Z10[1,0]-Z10[0,0]
dz11 = Z11[1,0]-Z11[0,0]

nom = np.trapz(np.trapz(abs(E11-E10)**2, dx=dx, axis=1), dx=dz, axis=0)
den = np.trapz(np.trapz(abs(E11)**2, dx=dx, axis=1), dx=dz, axis=0)
den2 = np.trapz(np.trapz(abs(E10)**2, dx=dx, axis=1), dx=dz, axis=0)
print(nom)
print(den)
print(den2)
print(nom/den)