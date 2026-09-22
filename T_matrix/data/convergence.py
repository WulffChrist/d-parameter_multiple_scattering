import numpy as np
import matplotlib.pyplot as plt

plt.figure()
lmax=8
for i in range(4):
    
    omega, Cext = np.loadtxt(fr'T_matrix\data\dimer_Q_dpdp_kx_Ez_Qs_wp4.0eV_gamma0.1eV_epsb1_R16nm_dg3nm_l{lmax}.txt', unpack=True)
    plt.plot(omega,Cext,label=fr'lmax = {lmax}')
    
    lmax+=1
plt.legend()
plt.show()