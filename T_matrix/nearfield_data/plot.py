import numpy as np
import matplotlib.pyplot as plt

E = np.loadtxt(fr'T_matrix\nearfield_data\dimer_Q_kx_Ez_E_omega3.25_R14R212nm_dg3nm_l10_res40.txt', unpack=True)
X = np.loadtxt(fr'T_matrix\nearfield_data\dimer_Q_kx_Ez_X_omega3.25_R14R212nm_dg3nm_l10_res40.txt', unpack=True)
Z = np.loadtxt(fr'T_matrix\nearfield_data\dimer_Q_kx_Ez_Z_omega3.25_R14R212nm_dg3nm_l10_res40.txt', unpack=True)

Rs = [4, 12]
gap = 3

fd = ['Ag','Na']

d = Rs[0]+Rs[1]+gap
pos = np.array([[-d/2,0,0],[d/2,0,0]]) # dimer
N = np.size(pos[:,0])

plt.figure()
plt.pcolor(X,Z,E)
plt.set_cmap('turbo')
plt.colorbar()
colors = ['pink','green']
for i in range(N):
    circle1 = plt.Circle((pos[i,0], pos[i,2]), Rs[i], color=colors[i], fill=True,linewidth=1)
    plt.gca().add_patch(circle1)
plt.xlabel("z")
plt.ylabel("x")
plt.show()