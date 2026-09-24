import numpy as np
import matplotlib.pyplot as plt

lmax=11

omega1, CextC = np.loadtxt(fr'T_matrix\data\dimer_C_kx_Ez_Qs_wp4.0eV_gamma0.1eV_epsb1_R16nm_dg3nm_l12.txt', unpack=True)
omega2, CextQ = np.loadtxt(fr'T_matrix\data\dimer_Q_dmdp_kx_Ez_Qs_wp4.0eV_gamma0.1eV_epsb1_R16nm_dg3nm_l{lmax}.txt', unpack=True)
omega3, CextQpp = np.loadtxt(fr'T_matrix\data\dimer_Q_dpdp_kx_Ez_Qs_wp4.0eV_gamma0.1eV_epsb1_R16nm_dg3nm_l{lmax}.txt', unpack=True)
omega4, CextQmm = np.loadtxt(fr'T_matrix\data\dimer_Q_dmdm_kx_Ez_Qs_wp4.0eV_gamma0.1eV_epsb1_R16nm_dg3nm_l{lmax}.txt', unpack=True)

plt.figure()
plt.plot(omega1,CextC,label='LRA')
plt.plot(omega2,CextQ,label='d+, d-',linestyle='dashed')
plt.plot(omega3,CextQpp,label='d+, d+')
plt.plot(omega4,CextQmm,label='d-, d-')
plt.xlabel(r'Energy [eV]')
plt.ylabel(r'$\sigma_{ext}$')
plt.legend()
plt.show()