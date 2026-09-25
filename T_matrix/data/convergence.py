import numpy as np
import matplotlib.pyplot as plt

lmax=11
mode = 'Q_dmdp'

plt.figure()
for i in range(1):
        
    omega, Cext = np.loadtxt(fr'T_matrix\data\dimer_{mode}_kx_Ez_Qs_wp4.0eV_gamma0.1eV_epsb1_R16nm_dg3nm_l{lmax}.txt', unpack=True)
    
    if i>0:
        nom = np.trapz(abs(Cext-Cext_1)**2,omega)
        den = np.trapz(abs(Cext)**2,omega)
        
        print(lmax,' ',nom/den)
    
    plt.plot(omega,Cext,label=fr'lmax = {lmax}')
    Cext_1 = Cext
    lmax+=1
plt.title(mode)
plt.legend()
plt.show()