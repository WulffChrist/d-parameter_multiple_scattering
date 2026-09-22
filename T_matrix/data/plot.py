import numpy as np
import matplotlib.pyplot as plt

omega, CextC = np.loadtxt(fr'T_matrix\data\dimer_C_kx_Ez_Qs_wp4.0eV_gamma0.1eV_epsb1_R16nm_dg3nm_l9.txt', unpack=True)
_, CextQ = np.loadtxt(fr'T_matrix\data\pmdimer_Q_kx_Ez_Qs_wp4.0eV_gamma0.1eV_epsb1_R16nm_dg3nm_l9.txt', unpack=True)
_, CextQpp = np.loadtxt(fr'T_matrix\data\ppdimer_Q_kx_Ez_Qs_wp4.0eV_gamma0.1eV_epsb1_R16nm_dg3nm_l9.txt', unpack=True)
_, CextQmm = np.loadtxt(fr'T_matrix\data\mmdimer_Q_kx_Ez_Qs_wp4.0eV_gamma0.1eV_epsb1_R16nm_dg3nm_l9.txt', unpack=True)

plt.figure()
plt.plot(omega,CextC,label='LRA')
plt.plot(omega,CextQ,label='d+, d-',linestyle='dashed')
plt.plot(omega,CextQpp,label='d+, d+')
plt.plot(omega,CextQmm,label='d-, d-')
plt.xlabel(r'Energy [eV]')
plt.ylabel(r'$\sigma_{ext}$')
plt.legend()
plt.show()